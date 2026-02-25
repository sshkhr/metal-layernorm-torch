#include <metal_stdlib>
using namespace metal;

// K3: SIMD group reductions — replace tree reduction with simd_sum().
//
// simd_sum() collapses an entire 32-lane SIMD group in one hardware call,
// replacing log2(32) = 5 shuffle-down steps.  Threadgroup memory is only
// used for the cross-SIMD-group step (max 32 floats), cutting barrier
// count from log2(ntg) to just 2 per reduction.
// Profiler expectation: barrier overhead drops, kernel becomes memory-BW-bound.

kernel void layernorm_simd(
    device const float*  src   [[buffer(0)]],
    device float*        dst   [[buffer(1)]],
    device const float*  gamma [[buffer(2)]],
    device const float*  beta  [[buffer(3)]],
    constant int64_t&    N     [[buffer(4)]],
    constant float&      eps   [[buffer(5)]],
    threadgroup float*   buf   [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint   ntg [[threads_per_threadgroup]])
{
    device const float* row_src = src + tgpig * N;

    // Pass 1: coalesced scalar loads, accumulate partial sum
    float local_sum = 0.0f;
    for (uint i = tpitg; i < uint(N); i += ntg)
        local_sum += row_src[i];

    // Two-level reduction: simd_sum within each SIMD group, then
    // threadgroup memory to communicate across SIMD groups.
    float sum = simd_sum(local_sum);
    if (ntg > 32) {
        if (sgitg == 0) buf[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) buf[sgitg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = buf[tiisg];
        sum = simd_sum(sum);
    }
    float mean = sum / float(N);

    // Pass 2: variance
    float local_var = 0.0f;
    for (uint i = tpitg; i < uint(N); i += ntg) {
        float diff = row_src[i] - mean;
        local_var += diff * diff;
    }

    float var = simd_sum(local_var);
    if (ntg > 32) {
        if (sgitg == 0) buf[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) buf[sgitg] = var;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        var = buf[tiisg];
        var = simd_sum(var);
    }
    float scale = rsqrt(var / float(N) + eps);

    // Pass 3: normalize with affine transform
    device float* row_dst = dst + tgpig * N;
    for (uint i = tpitg; i < uint(N); i += ntg)
        row_dst[i] = (row_src[i] - mean) * scale * gamma[i] + beta[i];
}
