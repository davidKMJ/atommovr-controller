/* STREAM mode: render just ahead of the card's read pointer. */

#ifndef AWG_ENGINE_STREAM_CUH
#define AWG_ENGINE_STREAM_CUH

#include "render.cuh"

#include "../../scapp/c_header/dlltyp.h"
#include "../../scapp/c_header/regs.h"
#include "../../scapp/c_header/spcerr.h"
#include "../../scapp/c_header/spcm_drv.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <mutex>

typedef struct AwgStreamCtx {
    drv_handle hCard;
    void* ring;                 /* GPU-resident SCAPP RDMA buffer */
    int64_t ring_frames;
    int32_t notify_frames;
    int32_t fill_start_threshold_promille;
    int64_t max_render_frames;  /* bound one launch so it cannot hog the pump */
    int cuda_device_index;
    float max_value;
    const AwgDeviceSchedule* sched;
    cudaStream_t stream;

    std::atomic<bool> stop_flag;
    bool card_started;
    bool start_failed;
    std::mutex start_mutex;
    std::condition_variable start_cv;

    /* Optional engine latch: written on the first failure so last_error is
     * visible while play() is still in flight. */
    std::mutex* error_mutex;
    char* last_error;
    size_t last_error_len;

    int failed;
    char err[512];
} AwgStreamCtx;

/* Latch the first failure. Print only if play() has already returned (the
 * card has started): startup failures are raised by the play() caller. */
static inline void awg_stream_report(AwgStreamCtx* c, const char* fmt, ...) {
    if (c->failed) {
        c->stop_flag.store(true, std::memory_order_relaxed);
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(c->err, sizeof(c->err), fmt, args);
    va_end(args);
    c->failed = 1;
    c->stop_flag.store(true, std::memory_order_relaxed);
    if (c->last_error != nullptr && c->error_mutex != nullptr && c->last_error_len > 0) {
        std::lock_guard<std::mutex> lock(*c->error_mutex);
        snprintf(c->last_error, c->last_error_len, "%s", c->err);
    }
    if (c->card_started) {
        fprintf(stderr, "[awg_engine] error: %s\n", c->err);
    }
}

static inline void awg_stream_fail_spc(AwgStreamCtx* c, const char* what) {
    uint32_t reg = 0;
    int32_t val = 0;
    char text[ERRORTEXTLEN] = {0};
    if (c->hCard != nullptr) {
        spcm_dwGetErrorInfo_i32(c->hCard, &reg, &val, text);
    }
    awg_stream_report(c, "%s failed: %s (register=0x%x value=%d)", what, text, reg, val);
}

static inline void awg_stream_pump(AwgStreamCtx* c) {
    /* The CUDA current device is per-thread: a fresh thread defaults to
     * device 0 whatever open() selected, and would render into a buffer it
     * does not own. */
    const cudaError_t setdev = cudaSetDevice(c->cuda_device_index);

    const int32_t frame_bytes = 4;
    const int32_t notify_bytes = c->notify_frames * frame_bytes;
    const int64_t ring_bytes = c->ring_frames * (int64_t)frame_bytes;
    const double fs = c->sched->sample_rate_hz;

    int64_t cursor = 0; /* absolute sample index into the round */

    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&t0] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    if (setdev != cudaSuccess) {
        awg_stream_report(c, "cudaSetDevice failed: %s", cudaGetErrorString(setdev));
    }

    while (!c->stop_flag.load(std::memory_order_relaxed)) {
        uint32_t dwErr = spcm_dwSetParam_i32(c->hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA);
        if (dwErr == ERR_TIMEOUT) {
            continue;
        }
        if (dwErr != ERR_OK) {
            uint32_t reg = 0;
            int32_t val = 0;
            char text[ERRORTEXTLEN] = {0};
            spcm_dwGetErrorInfo_i32(c->hCard, &reg, &val, text);
            awg_stream_report(c,
                              "M2CMD_DATA_WAITDMA failed after producing %.6f s (sample %lld) "
                              "in %.3f s wall; round is %.6f s: %s (register=0x%x value=%d)",
                              cursor / fs, (long long)cursor, elapsed(),
                              c->sched->total_samples / fs, text, reg, val);
            break;
        }

        int32_t avail_bytes = 0, user_pos_bytes = 0;
        spcm_dwGetParam_i32(c->hCard, SPC_DATA_AVAIL_USER_LEN, &avail_bytes);
        spcm_dwGetParam_i32(c->hCard, SPC_DATA_AVAIL_USER_POS, &user_pos_bytes);

        while (avail_bytes >= notify_bytes && !c->stop_flag.load(std::memory_order_relaxed)) {
            int64_t n = avail_bytes / frame_bytes;
            const int64_t to_wrap = c->ring_frames - user_pos_bytes / frame_bytes;
            if (to_wrap < n) n = to_wrap;               /* keep the launch contiguous */
            if (c->max_render_frames < n) n = c->max_render_frames;
            n = (n / c->notify_frames) * c->notify_frames;  /* whole chunks only */
            if (n <= 0) {
                break;
            }

            cudaError_t cerr = awg_render_span(c->sched, cursor, n, c->max_value,
                                               (int16_t*)((char*)c->ring + user_pos_bytes),
                                               c->stream);
            if (cerr == cudaSuccess) {
                cerr = cudaStreamSynchronize(c->stream);
            }
            if (cerr != cudaSuccess) {
                awg_stream_report(c, "render failed: %s", cudaGetErrorString(cerr));
                break;
            }

            const int32_t handed = (int32_t)(n * frame_bytes);
            dwErr = spcm_dwSetParam_i32(c->hCard, SPC_DATA_AVAIL_CARD_LEN, handed);
            if (dwErr != ERR_OK) {
                awg_stream_fail_spc(c, "SPC_DATA_AVAIL_CARD_LEN");
                break;
            }
            cursor += n;
            avail_bytes -= handed;
            user_pos_bytes = (int32_t)((user_pos_bytes + handed) % ring_bytes);
        }

        if (!c->card_started && !c->stop_flag.load(std::memory_order_relaxed)) {
            int32_t fill_promille = 0;
            spcm_dwGetParam_i32(c->hCard, SPC_FILLSIZEPROMILLE, &fill_promille);
            if (fill_promille > c->fill_start_threshold_promille) {
                dwErr = spcm_dwSetParam_i32(c->hCard, SPC_M2CMD,
                                            M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
                {
                    std::lock_guard<std::mutex> lock(c->start_mutex);
                    if (dwErr != ERR_OK) {
                        awg_stream_fail_spc(c, "M2CMD_CARD_START");
                        c->start_failed = true;
                    }
                    c->card_started = true;
                }
                c->start_cv.notify_all();
                if (dwErr != ERR_OK) {
                    break;
                }
            }
        }
    }

    /* produced < wall means the render or the link couldn't keep up;
     * produced == wall means the card was fed fine and stopped anyway. */
    fprintf(stderr, "[awg_engine] info: pump exiting: produced %.6f s in %.3f s wall\n",
            cursor / fs, elapsed());
    {
        /* never leave play() blocked on a pump that has exited */
        std::lock_guard<std::mutex> lock(c->start_mutex);
        if (!c->card_started) {
            c->start_failed = true;
            c->card_started = true;
        }
    }
    c->start_cv.notify_all();
}

#endif /* AWG_ENGINE_STREAM_CUH */
