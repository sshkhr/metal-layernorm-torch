#include <metal_stdlib>
using namespace metal;

// K1: Naive — one thread per row, three serial passes over device memory.
//
// Each thread independently reads an entire row three times (mean, variance,
// normalize). Adjacent threads in the same SIMD group read from different
// rows, so memory accesses are strided by N — defeating coalescing.
// Profiler expectation: very low GPU utilization, poor memory bandwidth.

kernel void layernorm_naive(
    device const float*  src   [[buffer(0)]],
    device float*        dst   [[buffer(1)]],
    device const float*  gamma [[buffer(2)]],
    device const float*  beta  [[buffer(3)]],
    constant int64_t&    N     [[buffer(4)]],
    constant float&      eps   [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    // Pass 1: compute mean
    float sum = 0.0f;
    for (uint i = 0; i < uint(N); i++)
        sum += src[row * N + i];
    float mean = sum / float(N);

    // Pass 2: compute variance
    float var = 0.0f;
    for (uint i = 0; i < uint(N); i++) {
        float diff = src[row * N + i] - mean;
        var += diff * diff;
    }
    float scale = rsqrt(var / float(N) + eps);

    // Pass 3: normalize with affine transform
    for (uint i = 0; i < uint(N); i++)
        dst[row * N + i] = (src[row * N + i] - mean) * scale * gamma[i] + beta[i];
}
