#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <unordered_map>
#include <string>

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// ---------------------------------------------------------------------------
// Pipeline state cache — compile each (shader_path, kernel_name) pair once.
//
// Without caching, every call to layernorm_forward() reads the .metal file
// from disk, compiles the shader, and creates a new PSO.  Metal shader
// compilation takes milliseconds, creating massive pipeline bubbles: the
// GPU finishes a kernel in microseconds then sits idle while the CPU
// compiles the next one.  GPU caches go completely cold between iterations.
// ---------------------------------------------------------------------------
struct PipelineCache {
    std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
    id<MTLDevice> device = nil;

    id<MTLDevice> getDevice() {
        if (!device) device = MTLCreateSystemDefaultDevice();
        return device;
    }

    id<MTLComputePipelineState> get(
        const std::string& shader_path,
        const std::string& kernel_name
    ) {
        std::string key = shader_path + "::" + kernel_name;
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;

        id<MTLDevice> dev = getDevice();
        NSError* error = nil;

        NSString* src = [NSString stringWithContentsOfFile:
            [NSString stringWithUTF8String:shader_path.c_str()]
            encoding:NSUTF8StringEncoding error:&error];
        TORCH_CHECK(src, "Failed to load shader: ",
                    error.localizedDescription.UTF8String);

        id<MTLLibrary> lib = [dev newLibraryWithSource:src
                                                options:nil error:&error];
        TORCH_CHECK(lib, "Failed to compile: ",
                    error.localizedDescription.UTF8String);

        NSString* fnName = [NSString stringWithUTF8String:kernel_name.c_str()];
        id<MTLFunction> fn = [lib newFunctionWithName:fnName];
        TORCH_CHECK(fn, "Kernel function '", kernel_name, "' not found");

        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&error];
        TORCH_CHECK(pso, "Pipeline creation failed: ",
                    error.localizedDescription.UTF8String);

        cache[key] = pso;
        return pso;
    }
};

static PipelineCache& getPipelineCache() {
    static PipelineCache instance;
    return instance;
}

torch::Tensor layernorm_forward(
    const torch::Tensor& input,    // [B, N] on MPS device
    const torch::Tensor& gamma,    // [N]
    const torch::Tensor& beta,     // [N]
    float eps,
    const std::string& shader_path,
    const std::string& kernel_name
) {
    TORCH_CHECK(input.device().is_mps(), "input must be an MPS tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int64_t B = input.size(0);      // batch (rows)
    int64_t N = input.size(1);      // hidden dim (columns)
    torch::Tensor output = torch::empty_like(input);

    // Resolve Metal function name (ablation variants share the same shader
    // function but use different dispatch configs).
    std::string metal_fn = kernel_name;
    if (kernel_name == "layernorm_naive_32") {
        metal_fn = "layernorm_naive";
    }

    // Look up (or compile once) the pipeline state
    id<MTLComputePipelineState> pso =
        getPipelineCache().get(shader_path, metal_fn);

    @autoreleasepool {
        // Integrate with PyTorch's MPS command system
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf, "Failed to get command buffer");
        dispatch_queue_t queue = torch::mps::get_dispatch_queue();

        dispatch_sync(queue, ^(){
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

            [enc setComputePipelineState:pso];
            [enc setBuffer:getMTLBufferStorage(input)
                    offset:input.storage_offset() * input.element_size()
                   atIndex:0];
            [enc setBuffer:getMTLBufferStorage(output)
                    offset:output.storage_offset() * output.element_size()
                   atIndex:1];
            [enc setBuffer:getMTLBufferStorage(gamma)
                    offset:0 atIndex:2];
            [enc setBuffer:getMTLBufferStorage(beta)
                    offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(int64_t) atIndex:4];
            [enc setBytes:&eps length:sizeof(float) atIndex:5];

            if (kernel_name == "layernorm_naive" ||
                kernel_name == "layernorm_naive_32") {
                // K1: One thread per row — use dispatchThreads so each
                // thread gets a unique row via [[thread_position_in_grid]].
                // No threadgroup memory needed.
                NSUInteger maxTg = (kernel_name == "layernorm_naive_32")
                    ? (NSUInteger)32 : (NSUInteger)1024;
                MTLSize gridSize = MTLSizeMake(B, 1, 1);
                NSUInteger tgWidth = std::min(
                    (NSUInteger)B, pso.maxTotalThreadsPerThreadgroup);
                tgWidth = std::min(tgWidth, maxTg);
                MTLSize tgSize = MTLSizeMake(tgWidth, 1, 1);
                [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

            } else if (kernel_name == "layernorm_shared") {
                // K2: One threadgroup per row, tree reduction in shared mem.
                // Tree reduction requires threadgroup size to be power of 2.
                NSUInteger tgWidth = std::min(
                    (NSUInteger)N, pso.maxTotalThreadsPerThreadgroup);
                tgWidth = std::min(tgWidth, (NSUInteger)1024);
                // Round down to nearest power of 2
                tgWidth = 1ULL << (63 - __builtin_clzll(tgWidth));
                // Threadgroup memory: one float per thread for tree reduction
                [enc setThreadgroupMemoryLength:tgWidth * sizeof(float)
                                        atIndex:0];
                MTLSize numGroups = MTLSizeMake(B, 1, 1);
                MTLSize tgSize = MTLSizeMake(tgWidth, 1, 1);
                [enc dispatchThreadgroups:numGroups
                    threadsPerThreadgroup:tgSize];

            } else if (kernel_name == "layernorm_simd") {
                // K3: Scalar coalesced loads — wants N threads per row
                // so each thread handles one element per iteration.
                NSUInteger tgWidth = std::min(
                    (NSUInteger)N, pso.maxTotalThreadsPerThreadgroup);
                tgWidth = std::min(tgWidth, (NSUInteger)1024);
                [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
                MTLSize numGroups = MTLSizeMake(B, 1, 1);
                MTLSize tgSize = MTLSizeMake(tgWidth, 1, 1);
                [enc dispatchThreadgroups:numGroups
                    threadsPerThreadgroup:tgSize];

            } else {
                // K4 (vectorized), K5 (fused), K6 (robust), K7 (regtiled):
                // float4 loads — each thread processes 4 elements per
                // iteration, so we only need ceil(N/4) threads per row.
                // Launching N threads wastes 75% of them at barriers.
                constexpr NSUInteger N_READS = 4;
                NSUInteger tgWidth = ((NSUInteger)N + N_READS - 1) / N_READS;
                tgWidth = std::min(tgWidth, pso.maxTotalThreadsPerThreadgroup);
                tgWidth = std::min(tgWidth, (NSUInteger)1024);
                [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
                MTLSize numGroups = MTLSizeMake(B, 1, 1);
                MTLSize tgSize = MTLSizeMake(tgWidth, 1, 1);
                [enc dispatchThreadgroups:numGroups
                    threadsPerThreadgroup:tgSize];
            }

            [enc endEncoding];
            torch::mps::commit();
        });
    }
    return output;
}

// ---------------------------------------------------------------------------
// Programmatic Metal GPU capture (requires METAL_CAPTURE_ENABLED=1 env var)
// ---------------------------------------------------------------------------

void start_gpu_capture(const std::string& output_path) {
    id<MTLDevice> device = getPipelineCache().getDevice();
    MTLCaptureManager* mgr = [MTLCaptureManager sharedCaptureManager];

    if (![mgr supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
        TORCH_CHECK(false,
            "GPU trace capture not supported. "
            "Set METAL_CAPTURE_ENABLED=1 before launching Python.");
    }

    // Remove any existing trace at the output path
    NSString* nsPath = [NSString stringWithUTF8String:output_path.c_str()];
    [[NSFileManager defaultManager] removeItemAtPath:nsPath error:nil];

    MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
    desc.captureObject = device;
    desc.destination = MTLCaptureDestinationGPUTraceDocument;
    desc.outputURL = [NSURL fileURLWithPath:nsPath];

    NSError* error = nil;
    BOOL ok = [mgr startCaptureWithDescriptor:desc error:&error];
    TORCH_CHECK(ok, "Failed to start GPU capture: ",
                error.localizedDescription.UTF8String);
}

void stop_gpu_capture() {
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward,
          "LayerNorm forward pass on Metal",
          py::arg("input"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps"), py::arg("shader_path"), py::arg("kernel_name"));

    m.def("start_gpu_capture", &start_gpu_capture,
          "Start a Metal GPU trace capture to a .gputrace file",
          py::arg("output_path"));
    m.def("stop_gpu_capture", &stop_gpu_capture,
          "Stop the active Metal GPU trace capture");
}
