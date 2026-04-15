#include <cuda_runtime.h>

#if defined(_WIN32)
#define CUDA_HELPER_EXPORT extern "C" __declspec(dllexport)
#else
#define CUDA_HELPER_EXPORT extern "C"
#endif

extern "C" __global__ void window_and_pack_kernel_f32(
    const float* audio,
    const float* window,
    float* output,
    int n_fft,
    int hop_length,
    int n_frames
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = n_fft * n_frames;
    if (idx >= total) {
        return;
    }

    const int frame = idx / n_fft;
    const int bin = idx - frame * n_fft;
    output[idx] = audio[frame * hop_length + bin] * window[bin];
}

CUDA_HELPER_EXPORT int launch_window_and_pack_f32(
    const float* audio,
    const float* window,
    float* output,
    int n_fft,
    int hop_length,
    int n_frames,
    void* stream
) {
    if (audio == nullptr || window == nullptr || output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_fft <= 0 || hop_length <= 0 || n_frames <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const int total = n_fft * n_frames;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    window_and_pack_kernel_f32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        audio,
        window,
        output,
        n_fft,
        hop_length,
        n_frames
    );

    return static_cast<int>(cudaGetLastError());
}

extern "C" __global__ void window_and_pack_batch_kernel_f32(
    const float* audio,
    const float* window,
    float* output,
    int n_samples,
    int n_fft,
    int hop_length,
    int n_frames,
    int n_channels
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = n_channels * n_frames * n_fft;
    if (idx >= total) {
        return;
    }

    const int frame_bin = idx % n_fft;
    const int frame_idx = (idx / n_fft) % n_frames;
    const int ch = idx / (n_fft * n_frames);

    const int audio_base = ch * n_samples + frame_idx * hop_length;
    output[idx] = audio[audio_base + frame_bin] * window[frame_bin];
}

CUDA_HELPER_EXPORT int launch_window_and_pack_batch_f32(
    const float* audio,
    const float* window,
    float* output,
    int n_samples,
    int n_fft,
    int hop_length,
    int n_frames,
    int n_channels,
    void* stream
) {
    if (audio == nullptr || window == nullptr || output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (
        n_samples <= 0 || n_fft <= 0 || hop_length <= 0 || n_frames <= 0
        || n_channels <= 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const int total = n_channels * n_frames * n_fft;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    window_and_pack_batch_kernel_f32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        audio,
        window,
        output,
        n_samples,
        n_fft,
        hop_length,
        n_frames,
        n_channels
    );

    return static_cast<int>(cudaGetLastError());
}

extern "C" __global__ void mel_project_power_kernel_f32(
    const float2* stft,
    const float* mel_basis,
    float* output,
    int n_bins,
    int n_frames,
    int n_mels
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = n_mels * n_frames;
    if (idx >= total) {
        return;
    }

    const int mel = idx / n_frames;
    const int frame = idx - mel * n_frames;

    float acc = 0.0f;
    const int mel_offset = mel * n_bins;
    const int frame_offset = frame * n_bins;
    for (int bin = 0; bin < n_bins; ++bin) {
        const float2 z = stft[frame_offset + bin];
        acc += mel_basis[mel_offset + bin] * (z.x * z.x + z.y * z.y);
    }

    output[idx] = acc;
}

CUDA_HELPER_EXPORT int launch_mel_project_power_f32(
    const void* stft,
    const float* mel_basis,
    float* output,
    int n_bins,
    int n_frames,
    int n_mels,
    void* stream
) {
    if (stft == nullptr || mel_basis == nullptr || output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_bins <= 0 || n_frames <= 0 || n_mels <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const int total = n_mels * n_frames;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    mel_project_power_kernel_f32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float2*>(stft),
        mel_basis,
        output,
        n_bins,
        n_frames,
        n_mels
    );

    return static_cast<int>(cudaGetLastError());
}

extern "C" __global__ void mel_project_power_batch_kernel_f32(
    const float2* stft,
    const float* mel_basis,
    float* output,
    int n_bins,
    int n_frames,
    int n_mels,
    int n_channels
) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = n_channels * n_mels * n_frames;
    if (idx >= total) {
        return;
    }

    const int frame = idx % n_frames;
    const int mel = (idx / n_frames) % n_mels;
    const int ch = idx / (n_frames * n_mels);

    float acc = 0.0f;
    const int mel_offset = mel * n_bins;
    const int stft_frame_offset = (ch * n_frames + frame) * n_bins;
    for (int bin = 0; bin < n_bins; ++bin) {
        const float2 z = stft[stft_frame_offset + bin];
        acc += mel_basis[mel_offset + bin] * (z.x * z.x + z.y * z.y);
    }

    output[idx] = acc;
}

CUDA_HELPER_EXPORT int launch_mel_project_power_batch_f32(
    const void* stft,
    const float* mel_basis,
    float* output,
    int n_bins,
    int n_frames,
    int n_mels,
    int n_channels,
    void* stream
) {
    if (stft == nullptr || mel_basis == nullptr || output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_bins <= 0 || n_frames <= 0 || n_mels <= 0 || n_channels <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const int total = n_channels * n_mels * n_frames;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    mel_project_power_batch_kernel_f32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float2*>(stft),
        mel_basis,
        output,
        n_bins,
        n_frames,
        n_mels,
        n_channels
    );

    return static_cast<int>(cudaGetLastError());
}

