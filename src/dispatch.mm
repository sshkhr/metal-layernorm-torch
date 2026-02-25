#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
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

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError* error = nil;

        // Load and compile the Metal shader at runtime
        NSString* src = [NSString stringWithContentsOfFile:
            [NSString stringWithUTF8String:shader_path.c_str()]
            encoding:NSUTF8StringEncoding error:&error];
        TORCH_CHECK(src, "Failed to load shader: ",
                    error.localizedDescription.UTF8String);

        id<MTLLibrary> lib = [device newLibraryWithSource:src
                                                  options:nil error:&error];
        TORCH_CHECK(lib, "Failed to compile: ",
                    error.localizedDescription.UTF8String);

        NSString* fnName = [NSString stringWithUTF8String:kernel_name.c_str()];
        id<MTLFunction> fn = [lib newFunctionWithName:fnName];
        TORCH_CHECK(fn, "Kernel function '", kernel_name, "' not found");

        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:fn error:&error];
        TORCH_CHECK(pso, "Pipeline creation failed: ",
                    error.localizedDescription.UTF8String);

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

            if (kernel_name == "layernorm_naive") {
                // K1: One thread per row — use dispatchThreads so each
                // thread gets a unique row via [[thread_position_in_grid]].
                // No threadgroup memory needed.
                MTLSize gridSize = MTLSizeMake(B, 1, 1);
                NSUInteger tgWidth = std::min(
                    (NSUInteger)B, pso.maxTotalThreadsPerThreadgroup);
                tgWidth = std::min(tgWidth, (NSUInteger)256);
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

            } else {
                // K3 (simd) and K4 (vectorized): SIMD-based reduction.
                // Only 32 floats of threadgroup memory for cross-SIMD step.
                NSUInteger tgWidth = std::min(
                    (NSUInteger)N, pso.maxTotalThreadsPerThreadgroup);
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
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
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
