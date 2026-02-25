#include <metal_stdlib>
using namespace metal;

// K7: Register-tiled — single device memory read of X.
//
// During the fused stats pass each thread caches its float4 chunks in a
// thread-local array.  The normalize pass then reads from this register
// cache instead of re-reading X from device memory, cutting total traffic
// from 20N bytes (K5/K6) to 16N bytes.
//
// For the typical single_row regime (N ≤ 4096 with 1024 threads) each
// thread caches exactly one float4 and the compiler can elide the array.
// For larger N the looped variant stores up to MAX_ITERS float4s per
// thread — feasible up to N = 32768 without register spilling on Apple
// Silicon's generous register files.
//
// Profiler expectation: ~20% faster than K6.  If register spilling occurs
// at high N, the GPU trace will show increased memory traffic — that is
// the signal to fall back to K6 for those shapes.

kernel void layernorm_regtiled(
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
    constexpr uint N_READS   = 4;
    constexpr uint MAX_ITERS = 8;   // supports N up to ntg * 4 * 8 = 32768

    device const float* row = src + tgpig * N;
    uint uN  = uint(N);
    uint n4  = uN / N_READS;
    uint rem = uN % N_READS;

    // ---- Pass 1: Load + Stats (single read from device memory) ----
    float4 cache[MAX_ITERS];
    uint num_iters = 0;

    float partial_sum    = 0.0f;
    float partial_sum_sq = 0.0f;

    for (uint r = 0; r < n4; r += ntg) {
        uint i = r + tpitg;
        if (i < n4) {
            uint base = i * N_READS;
            float4 v = float4(row[base], row[base + 1],
                              row[base + 2], row[base + 3]);
            cache[num_iters] = v;
            partial_sum    += v.x + v.y + v.z + v.w;
            partial_sum_sq += dot(v, v);
        }
        num_iters++;
    }

    // Tail: thread 0 handles remaining 1-3 elements (not cached —
    // these are cheap scalar loads in the normalize pass too)
    float tail_vals[3] = {0.0f, 0.0f, 0.0f};
    if (tpitg == 0) {
        uint base = n4 * N_READS;
        for (uint i = 0; i < rem; i++) {
            float v = row[base + i];
            tail_vals[i] = v;
            partial_sum    += v;
            partial_sum_sq += v * v;
        }
    }

    // ---- Reduce sum ----
    float sum = simd_sum(partial_sum);
    if (ntg > 32) {
        if (sgitg == 0) buf[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) buf[sgitg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = buf[tiisg];
        sum = simd_sum(sum);
    }
    float mean = sum / float(N);

    // ---- Reduce sum_sq ----
    float sum_sq = simd_sum(partial_sum_sq);
    if (ntg > 32) {
        if (sgitg == 0) buf[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) buf[sgitg] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum_sq = buf[tiisg];
        sum_sq = simd_sum(sum_sq);
    }
    float var = sum_sq / float(N) - mean * mean;
    var = var < 1e-6f ? 0.0f : var;
    float inv_std = metal::precise::rsqrt(var + eps);

    // ---- Pass 2: Normalize from register cache (NO re-read of src) ----
    device float* out = dst + tgpig * N;
    uint iter = 0;

    for (uint r = 0; r < n4; r += ntg) {
        uint i = r + tpitg;
        if (i < n4) {
            uint base = i * N_READS;
            float4 v = cache[iter];
            #pragma unroll
            for (uint j = 0; j < N_READS; j++) {
                float norm = (v[j] - mean) * inv_std;
                out[base + j] = fma(norm, gamma[base + j], beta[base + j]);
            }
        }
        iter++;
    }

    // Tail: thread 0 normalizes remaining elements from cached tail_vals
    if (tpitg == 0) {
        uint base = n4 * N_READS;
        for (uint i = 0; i < rem; i++) {
            float norm = (tail_vals[i] - mean) * inv_std;
            out[base + i] = fma(norm, gamma[base + i], beta[base + i]);
        }
    }
}
