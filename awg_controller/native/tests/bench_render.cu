/* Render-throughput benchmark. Needs a CUDA GPU, but NO card and no driver.
 *
 * Answers the only question that decides whether a given grid size can be
 * streamed in real time: how many tone-samples/s can this GPU render?
 *
 *   make bench && ./tests/bench_render [sample_rate_MS/s]
 *
 * The engine must sustain  sample_rate * 2 * tones_per_channel  tone-samples/s.
 * Anything below 1.0x realtime underruns no matter how the ring or
 * notify_samples are tuned -- the pump can only hand over what it has
 * rendered.
 */

#include "../render.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void build_round(std::vector<double>& durs, std::vector<int32_t>& counts,
                        std::vector<AWGRoundRamp>& ramps, int n_batches, int tones_per_ch,
                        double f_lo, double f_hi) {
    durs.assign((size_t)n_batches, 5e-6); /* experiment scale: 5 us moves */
    counts.assign((size_t)n_batches, 2 * tones_per_ch);
    ramps.clear();
    const double span = (f_hi - f_lo) / (tones_per_ch > 1 ? tones_per_ch - 1 : 1);
    for (int b = 0; b < n_batches; ++b) {
        for (int ch = 0; ch < 2; ++ch) {
            for (int t = 0; t < tones_per_ch; ++t) {
                AWGRoundRamp r;
                r.channel = ch;
                r.tone_index = t;
                r.f_start_hz = f_lo + span * t;
                /* every other batch actually moves, so the chirp path is exercised */
                r.f_end_hz = r.f_start_hz + ((b % 2) ? span * 0.5 : 0.0);
                r.amplitude_pct = 40.0 / tones_per_ch;
                r.phase_deg = 0.0;
                ramps.push_back(r);
            }
        }
    }
}

int main(int argc, char** argv) {
    const double fs = (argc > 1) ? atof(argv[1]) * 1e6 : 1250e6;
    const int n_batches = 500;
    const int64_t chunk = 262144; /* notify_samples */
    const int iters = 200;

    printf("render benchmark -- sample_rate = %.1f MS/s, %d batches of 5 us,\n"
           "chunk = %lld frames (%.1f us of waveform)\n\n",
           fs / 1e6, n_batches, (long long)chunk, chunk / fs * 1e6);
    printf("  %6s %14s %11s %12s %10s\n", "tones", "tone-samp/s", "us/chunk", "x realtime",
           "verdict");

    int16_t* dst = nullptr;
    if (cudaMalloc((void**)&dst, (size_t)chunk * 2 * sizeof(int16_t)) != cudaSuccess) {
        printf("cudaMalloc failed\n");
        return 1;
    }

    for (int tones : {1, 5, 10, 15, 20, 30}) {
        std::vector<double> durs;
        std::vector<int32_t> counts;
        std::vector<AWGRoundRamp> ramps;
        build_round(durs, counts, ramps, n_batches, tones, 60e6, 110e6);

        AwgSchedule sch;
        char err[256] = {0};
        if (awg_schedule_build(&sch, durs.data(), n_batches, ramps.data(),
                               (int32_t)ramps.size(), counts.data(), tones, tones, fs, 0, err,
                               sizeof(err)) != 0) {
            printf("  schedule build failed: %s\n", err);
            continue;
        }
        AwgDeviceSchedule dev = {};
        if (awg_device_schedule_upload(&dev, &sch) != cudaSuccess) {
            printf("  upload failed\n");
            awg_schedule_free(&sch);
            continue;
        }

        /* warm up, then time */
        awg_render_span(&dev, 0, chunk, 32767.0f, dst, 0);
        cudaDeviceSynchronize();

        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int k = 0; k < iters; ++k) {
            awg_render_span(&dev, (int64_t)k * chunk, chunk, 32767.0f, dst, 0);
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a);
        cudaEventDestroy(b);

        const double per_chunk_us = ms * 1000.0 / iters;
        const double ts_per_s = (double)chunk * 2 * tones * iters / (ms / 1000.0);
        const double realtime = ts_per_s / (fs * 2 * tones);
        printf("  %6d %14.2e %11.1f %12.2f %10s\n", tones, ts_per_s, per_chunk_us, realtime,
               realtime > 1.5 ? "OK" : (realtime > 1.0 ? "marginal" : "TOO SLOW"));

        awg_device_schedule_free(&dev);
        awg_schedule_free(&sch);
    }

    cudaFree(dst);
    printf("\n  need > 1.0x, and >1.5x for headroom against jitter.\n");
    return 0;
}
