// Metal FFT kernel for iron-librosa STFT/iSTFT
// Implements in-place radix-2 FFT for power-of-2 sizes
// Processes one frame per thread-group in parallel

#include <metal_stdlib>
using namespace metal;

struct Complex {
    float real;
    float imag;
};

// Bitwise reverse permutation for FFT index
uint bit_reverse(uint i, uint n_bits) {
    uint result = 0;
    for (uint b = 0; b < n_bits; b++) {
        result = (result << 1) | (i & 1);
        i >>= 1;
    }
    return result;
}

// Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
Complex complex_mul(Complex a, Complex b) {
    return Complex{
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

// Complex addition
Complex complex_add(Complex a, Complex b) {
    return Complex{a.real + b.real, a.imag + b.imag};
}

// Complex subtraction
Complex complex_sub(Complex a, Complex b) {
    return Complex{a.real - b.real, a.imag - b.imag};
}

// Compute twiddle factor: exp(-2πi * k / N) for forward FFT
Complex twiddle_forward(uint k, uint N) {
    float angle = -2.0f * M_PI_F * float(k) / float(N);
    return Complex{cos(angle), sin(angle)};
}

// Compute twiddle factor: exp(2πi * k / N) for inverse FFT
Complex twiddle_inverse(uint k, uint N) {
    float angle = 2.0f * M_PI_F * float(k) / float(N);
    return Complex{cos(angle), sin(angle)};
}

/// Forward FFT kernel (in-place, single frame)
/// data: input/output complex array (size N, must be power of 2)
/// N: FFT size (must be power of 2)
/// frame_idx: which frame to process (for batched operation)
/// all frames are independent; one frame per invocation
kernel void fft_forward_frame(
    device Complex* data [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    // Use group-local memory for scratch (allows fast bit-reversal shuffle)
    threadgroup Complex scratch[2048];  // Max 2048-point FFT

    uint local_id = tid;

    uint frame_base = gid * N;

    // Step 1: Bit-reversal permutation with threadgroup synchronization
    if (local_id < N) {
        uint rev_idx = bit_reverse(local_id, uint(log2(float(N))));
        scratch[rev_idx] = data[frame_base + local_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Radix-2 Cooley-Tukey FFT in-place
    for (uint stage = 1; stage <= uint(log2(float(N))); stage++) {
        uint m = (1u << stage);         // 2^stage
        uint m2 = m >> 1;               // m/2

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local_id < N) {
            uint k = local_id & (m2 - 1);
            uint j = ((local_id >> stage) << stage) + k;
            uint j2 = j + m2;

            if (j2 < N) {
                Complex w = twiddle_forward(k, m);
                Complex t = complex_mul(scratch[j2], w);
                Complex u = scratch[j];

                scratch[j] = complex_add(u, t);
                scratch[j2] = complex_sub(u, t);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Write back to device memory
    if (local_id < N) {
        data[frame_base + local_id] = scratch[local_id];
    }
}

/// Inverse FFT kernel (in-place, single frame)
/// Identical to forward but uses different twiddle factors and scaling
kernel void fft_inverse_frame(
    device Complex* data [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup Complex scratch[2048];

    uint local_id = tid;
    uint frame_base = gid * N;
    uint n_bits = uint(log2(float(N)));

    // Step 1: Bit-reversal
    if (local_id < N) {
        uint rev_idx = bit_reverse(local_id, n_bits);
        scratch[rev_idx] = data[frame_base + local_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Radix-2 FFT with inverse twiddle factors
    for (uint stage = 1; stage <= n_bits; stage++) {
        uint m = (1u << stage);
        uint m2 = m >> 1;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local_id < N) {
            uint k = local_id & (m2 - 1);
            uint j = ((local_id >> stage) << stage) + k;
            uint j2 = j + m2;

            if (j2 < N) {
                Complex w = twiddle_inverse(k, m);
                Complex t = complex_mul(scratch[j2], w);
                Complex u = scratch[j];

                scratch[j] = complex_add(u, t);
                scratch[j2] = complex_sub(u, t);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Write back (unscaled, matches rustfft inverse semantics)
    if (local_id < N) {
        data[frame_base + local_id] = scratch[local_id];
    }
}

