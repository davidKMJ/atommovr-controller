/* GPU rendering of a span of the round. */

#ifndef AWG_ENGINE_RENDER_CUH
#define AWG_ENGINE_RENDER_CUH

#include "schedule.h"

#include <cuda_runtime.h>

/* Device-resident mirror of AwgSchedule. */
typedef struct AwgDeviceSchedule {
    AwgSegment* segments;
    int64_t* batch_start;
    int32_t n_batches;
    int32_t n_tones[2];
    int32_t n_tones_total;
    int64_t total_samples;
    double sample_rate_hz;
} AwgDeviceSchedule;

static inline void awg_device_schedule_free(AwgDeviceSchedule* d) {
    if (d == NULL) {
        return;
    }
    if (d->segments) cudaFree(d->segments);
    if (d->batch_start) cudaFree(d->batch_start);
    d->segments = NULL;
    d->batch_start = NULL;
    d->n_batches = 0;
    d->total_samples = 0;
}

/* Copy a host schedule to the device. */
static inline cudaError_t awg_device_schedule_upload(AwgDeviceSchedule* d,
                                                     const AwgSchedule* h) {
    awg_device_schedule_free(d);
    d->n_batches = h->n_batches;
    d->n_tones[0] = h->n_tones[0];
    d->n_tones[1] = h->n_tones[1];
    d->n_tones_total = h->n_tones_total;
    d->total_samples = h->total_samples;
    d->sample_rate_hz = h->sample_rate_hz;

    const size_t seg_bytes = (size_t)h->n_batches * (size_t)h->n_tones_total * sizeof(AwgSegment);
    const size_t bs_bytes = ((size_t)h->n_batches + 1) * sizeof(int64_t);

    cudaError_t err = cudaMalloc((void**)&d->segments, seg_bytes);
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void**)&d->batch_start, bs_bytes);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(d->segments, h->segments, seg_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    return cudaMemcpy(d->batch_start, h->batch_start, bs_bytes, cudaMemcpyHostToDevice);
}

/* One thread per frame, emitting BOTH channels as a single 32-bit store. */
__global__ void awg_render_kernel(const AwgSegment* __restrict__ segments,
                                  const int64_t* __restrict__ batch_start, int32_t n_batches,
                                  int32_t n_tones_total, int32_t n_tones0, int32_t n_tones1,
                                  int64_t total_samples, int64_t abs_start, int64_t n_frames,
                                  float max_value, uint32_t* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_frames) {
        return;
    }
    const int64_t abs_sample = abs_start + i;
    /* one lookup per frame, shared by every tone on both channels */
    const int32_t b = awg_batch_at(batch_start, n_batches, total_samples, abs_sample);
    const AwgSegment* segs = segments + (int64_t)b * n_tones_total;

    float acc0 = 0.0f;
    for (int32_t t = 0; t < n_tones0; ++t) {
        const AwgSegment* seg = &segs[t];
        acc0 += __sinf(6.2831853071795864f *
                       awg_phase_cycles(awg_segment_total_phase_q64(seg, abs_sample))) *
                seg->amplitude;
    }
    float acc1 = 0.0f;
    for (int32_t t = 0; t < n_tones1; ++t) {
        const AwgSegment* seg = &segs[n_tones0 + t];
        acc1 += __sinf(6.2831853071795864f *
                       awg_phase_cycles(awg_segment_total_phase_q64(seg, abs_sample))) *
                seg->amplitude;
    }

    /* seg->amplitude is normalised (amplitude_pct/100); DAC scaling happens here. */
    const int32_t s0 = __float2int_rn(fminf(fmaxf(acc0, -1.0f), 1.0f) * max_value);
    const int32_t s1 = __float2int_rn(fminf(fmaxf(acc1, -1.0f), 1.0f) * max_value);
    out[i] = ((uint32_t)(uint16_t)(int16_t)s1 << 16) | (uint32_t)(uint16_t)(int16_t)s0;
}

/* Render n_frames into `dst` (contiguous, interleaved). Caller splits at the
 * ring wrap so this stays contiguous. */
static inline cudaError_t awg_render_span(const AwgDeviceSchedule* d, int64_t abs_start,
                                          int64_t n_frames, float max_value, int16_t* dst,
                                          cudaStream_t stream) {
    if (n_frames <= 0) {
        return cudaSuccess;
    }
    const int threads = 256;
    const int64_t blocks = (n_frames + threads - 1) / threads;
    awg_render_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        d->segments, d->batch_start, d->n_batches, d->n_tones_total, d->n_tones[0],
        d->n_tones[1], d->total_samples, abs_start, n_frames, max_value,
        reinterpret_cast<uint32_t*>(dst));
    return cudaGetLastError();
}

#endif /* AWG_ENGINE_RENDER_CUH */
