#include <metal_stdlib>
using namespace metal;

// K2: Threadgroup memory + coalesced access — one threadgroup per row.
//
// Threads cooperatively load the row with coalesced access (adjacent
// threads read adjacent elements), then reduce via a tree in threadgroup
// memory.  This needs log2(ntg) barriers per reduction — on Apple Silicon
// the threadgroup barrier cost is the bottleneck the profiler should reveal.

kernel void layernorm_shared(
    device const float*  src   [[buffer(0)]],
    device float*        dst   [[buffer(1)]],
    device const float*  gamma [[buffer(2)]],
    device const float*  beta  [[buffer(3)]],
    constant int64_t&    N     [[buffer(4)]],
    constant float&      eps   [[buffer(5)]],
    threadgroup float*   buf   [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tpitg [[thread_position_in_threadgroup]],
    uint   ntg [[threads_per_threadgroup]])
{
    device const float* row_src = src + tgpig * N;

    // Pass 1: coalesced loads, accumulate partial sums
    float local_sum = 0.0f;
    for (uint i = tpitg; i < uint(N); i += ntg)
        local_sum += row_src[i];

    // Tree reduction in threadgroup memory (requires ntg to be power of 2)
    buf[tpitg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg / 2; stride > 0; stride /= 2) {
        if (tpitg < stride)
            buf[tpitg] += buf[tpitg + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = buf[0] / float(N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: variance with tree reduction
    float local_var = 0.0f;
    for (uint i = tpitg; i < uint(N); i += ntg) {
        float diff = row_src[i] - mean;
        local_var += diff * diff;
    }

    buf[tpitg] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg / 2; stride > 0; stride /= 2) {
        if (tpitg < stride)
            buf[tpitg] += buf[tpitg + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(buf[0] / float(N) + eps);

    // Pass 3: normalize with affine transform (coalesced writes)
    device float* row_dst = dst + tgpig * N;
    for (uint i = tpitg; i < uint(N); i += ntg)
        row_dst[i] = (row_src[i] - mean) * scale * gamma[i] + beta[i];
}
