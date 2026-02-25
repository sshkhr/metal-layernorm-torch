#include <metal_stdlib>
using namespace metal;

// K4: Vectorized float4 loads + SIMD reductions.
//
// Builds on K3 by casting device pointers to float4*, loading 128 bits
// (4 floats) per thread per iteration.  This maximizes memory transaction
// efficiency on the memory-bound inner loops.  The two-level SIMD +
// threadgroup reduction is identical to K3.
// Profiler expectation: approaches peak memory bandwidth (~80-85%).

kernel void layernorm_vectorized(
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
    device const float4* x = (device const float4*)(src + tgpig * N);

    // Pass 1: compute mean via vectorized loads
    float4 sum4 = 0.0f;
    for (uint i = tpitg; i < N / 4; i += ntg)
        sum4 += x[i];
    float sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];

    // Two-level SIMD + threadgroup reduction
    sum = simd_sum(sum);
    if (ntg > 32) {
        if (sgitg == 0) buf[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) buf[sgitg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = buf[tiisg];
        sum = simd_sum(sum);
    }
    float mean = sum / float(N);

    // Pass 2: compute variance
    float4 var4 = 0.0f;
    for (uint i = tpitg; i < N / 4; i += ntg) {
        float4 diff = x[i] - mean;
        var4 += diff * diff;
    }
    float var = var4[0] + var4[1] + var4[2] + var4[3];

    var = simd_sum(var);
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
    device float4* y = (device float4*)(dst + tgpig * N);
    device const float4* g = (device const float4*)gamma;
    device const float4* b = (device const float4*)beta;
    for (uint i = tpitg; i < N / 4; i += ntg)
        y[i] = fma((x[i] - mean) * scale, g[i], b[i]);
}
