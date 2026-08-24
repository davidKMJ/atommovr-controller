/* STREAM mode: render just ahead of the card's read pointer.
 *
 * Memory is the RDMA ring alone, so round length is unbounded. The ring
 * absorbs *jitter* -- a stall shorter than its depth is ridden out -- but it
 * does not relax the average rate. In steady state WAITDMA wakes once per
 * notify chunk and this loop renders one chunk per wake, so each render must
 * finish within notify_samples/sample_rate. At 1.25 GS/s a 16384-frame chunk
 * is 13 us, about one kernel launch, and underruns; 262144 gives 210 us.
 * Size notify_samples for the render budget, the ring for hiccup tolerance.
 *
 * `cursor` is an absolute sample index that never resets. Past the end of the
 * round the schedule clamps to the final batch, so the engine parks on the
 * last frequencies phase-exactly, with no chunk replay and hence no seam.
 *
 * NB: sustaining this needs sample_rate*4 B/s over PCIe forever. An M4i is
 * Gen2 x8, so beyond roughly 500-800 MS/s two-channel the card outruns its
 * own link no matter how this is tuned -- use MEMORY mode instead.
 */

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
#include <mutex>
#include <stdio.h>

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
    std::atomic<bool> running;
    bool card_started;
    bool start_failed;
    std::mutex start_mutex;
    std::condition_variable start_cv;

    int failed;
    char err[256];
} AwgStreamCtx;

static inline void awg_stream_fail(AwgStreamCtx* c, const char* what, uint32_t code) {
    if (!c->failed) {
        snprintf(c->err, sizeof(c->err), "%s failed (spcm err %u)", what, code);
        c->failed = 1;
    }
    c->stop_flag.store(true, std::memory_order_relaxed);
}

static inline void awg_stream_pump(AwgStreamCtx* c) {
    /* The CUDA current device is per-thread: a fresh thread defaults to
     * device 0 whatever open() selected, and would render into a buffer it
     * does not own. */
    cudaSetDevice(c->cuda_device_index);

    const int32_t frame_bytes = 4;
    const int32_t notify_bytes = c->notify_frames * frame_bytes;
    const int64_t ring_bytes = c->ring_frames * (int64_t)frame_bytes;
    const double fs = c->sched->sample_rate_hz;

    int64_t cursor = 0; /* absolute sample index into the round */

    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&t0] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    while (!c->stop_flag.load(std::memory_order_relaxed)) {
        uint32_t dwErr = spcm_dwSetParam_i32(c->hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA);
        if (dwErr == ERR_TIMEOUT) {
            continue;
        }
        if (dwErr != ERR_OK) {
            fprintf(stderr,
                    "[awg_engine] WAITDMA failed after producing %.6f s (sample %lld) "
                    "in %.3f s wall; round is %.6f s\n",
                    cursor / fs, (long long)cursor, elapsed(),
                    c->sched->total_samples / fs);
            awg_stream_fail(c, "M2CMD_DATA_WAITDMA", dwErr);
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
                if (!c->failed) {
                    snprintf(c->err, sizeof(c->err), "render failed: %s",
                             cudaGetErrorString(cerr));
                    c->failed = 1;
                }
                c->stop_flag.store(true, std::memory_order_relaxed);
                break;
            }

            const int32_t handed = (int32_t)(n * frame_bytes);
            dwErr = spcm_dwSetParam_i32(c->hCard, SPC_DATA_AVAIL_CARD_LEN, handed);
            if (dwErr != ERR_OK) {
                awg_stream_fail(c, "SPC_DATA_AVAIL_CARD_LEN", dwErr);
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
                        awg_stream_fail(c, "M2CMD_CARD_START", dwErr);
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
    fprintf(stderr, "[awg_engine] pump exiting: produced %.6f s in %.3f s wall\n", cursor / fs,
            elapsed());
    c->running.store(false, std::memory_order_relaxed);
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
