#include <metal_stdlib>
using namespace metal;

// K5: Fused single-pass statistics + vectorized float4 loads.
//
// Fuses the mean and variance passes into a single loop using the identity
// Var(x) = E[x²] - E[x]².  Both partial_sum and partial_sum_sq are
// accumulated simultaneously, cutting device memory reads of X from 3 to 2
// (one for stats, one for normalize).
//
// Same float4 pointer cast and two-level SIMD reduction as K4.
// Profiler expectation: ~30-40% faster than K4 due to eliminated memory pass.
//
// Numerical note: E[x²] - E[x]² is less stable than the two-pass E[(x-μ)²]
// when variance is tiny relative to the mean.  For typical transformer
// activations (roughly zero-centered) this is a non-issue.

kernel void layernorm_fused(
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

    // ---- Fused Pass: accumulate sum and sum-of-squares simultaneously ----
    float4 sum4 = 0.0f;
    float4 sum_sq4 = 0.0f;
    for (uint i = tpitg; i < uint(N) / 4; i += ntg) {
        float4 v = x[i];
        sum4 += v;
        sum_sq4 += v * v;
    }
    float sum    = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    float sum_sq = sum_sq4[0] + sum_sq4[1] + sum_sq4[2] + sum_sq4[3];

    // ---- Reduce sum via two-level SIMD + threadgroup ----
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

    // ---- Reduce sum_sq via same pattern ----
    sum_sq = simd_sum(sum_sq);
    if (ntg > 32) {
        if (sgitg == 0) buf[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) buf[sgitg] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum_sq = buf[tiisg];
        sum_sq = simd_sum(sum_sq);
    }
    float var = sum_sq / float(N) - mean * mean;
    float scale = rsqrt(var + eps);

    // ---- Normalize with affine transform (second read of X) ----
    device float4* y = (device float4*)(dst + tgpig * N);
    device const float4* g = (device const float4*)gamma;
    device const float4* b = (device const float4*)beta;
    for (uint i = tpitg; i < uint(N) / 4; i += ntg)
        y[i] = fma((x[i] - mean) * scale, g[i], b[i]);
}
