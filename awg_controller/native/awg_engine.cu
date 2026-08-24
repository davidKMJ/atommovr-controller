/*
 * awg_engine.cu -- facade over the two replay modes (see awg_engine.h).
 *
 * Card setup, error handling and lifetime live here; the modes themselves are
 * stream.cuh (FIFO render-ahead) and sequence.cuh (precompute + card DRAM).
 * Phase maths is fixed point (phase.h), shared verbatim between the host-side
 * carry and the device kernel and unit-tested off-hardware against
 * awg_controller.scapp -- see `make test test-scapp`.
 */

#include "sequence.cuh"
#include "stream.cuh"

#include "../../scapp/cuda_rdma/common/spcm_cuda_common.h"

#include <cstdarg>
#include <cstdio>
#include <thread>

namespace {
constexpr double kMaxSafeOutputV = 2.0;
constexpr int kStartupTimeoutMs = 5000;
constexpr int64_t kDefaultHoldTailSamples = 1 << 20;
} // namespace

struct AWGEngine {
    drv_handle hCard = nullptr;
    void* dma_buffer = nullptr;   /* STREAM: the ring. MEMORY: upload staging. */

    int32_t mode = AWG_ENGINE_MODE_STREAM;
    int32_t notify_samples = 0;
    int64_t dma_buffer_samples = 0;
    int32_t fill_start_threshold_promille = 800;
    int64_t hold_tail_samples = kDefaultHoldTailSamples;
    double sample_rate_hz = 0.0;
    int16_t max_value = 1;
    int32_t n_tones[2] = {0, 0};
    int cuda_device_index = 0;
    cudaStream_t stream = nullptr;

    AwgSchedule schedule{};             /* the round */
    AwgDeviceSchedule dev_schedule{};
    AwgSchedule tail{};                 /* MEMORY: the looping park segment */
    AwgDeviceSchedule dev_tail{};
    bool have_round = false;

    AwgStreamCtx sctx{};                /* STREAM only */
    std::thread pump_thread;
    /* MEMORY has no pump to watch, so track playback here: uploading a new
     * round writes over card memory the sequencer is actively replaying. */
    bool playing = false;

    std::mutex error_mutex;
    char last_error[512] = {0};
};

namespace {

void set_error(AWGEngine* pc, const char* fmt, ...) {
    if (pc == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(pc->error_mutex);
    va_list args;
    va_start(args, fmt);
    vsnprintf(pc->last_error, sizeof(pc->last_error), fmt, args);
    va_end(args);
    fprintf(stderr, "[awg_engine] %s\n", pc->last_error);
}

bool check_spc(AWGEngine* pc, uint32_t dwErr, const char* what) {
    if (dwErr == ERR_OK) {
        return true;
    }
    /* For a validation command (WRITESETUP) that fails because of an
     * inconsistent combination of registers, `text` alone is usually the
     * generic "the setup isn't valid" -- register/value name the actual
     * offending register, which `text` does not. */
    uint32_t reg = 0;
    int32_t val = 0;
    char text[ERRORTEXTLEN] = {0};
    spcm_dwGetErrorInfo_i32(pc->hCard, &reg, &val, text);
    set_error(pc, "%s failed: %s (register=0x%x value=%d)", what, text, reg, val);
    return false;
}

bool check_cuda(AWGEngine* pc, cudaError_t err, const char* what) {
    if (err == cudaSuccess) {
        return true;
    }
    set_error(pc, "%s failed: %s", what, cudaGetErrorString(err));
    return false;
}

/* spcm latches the last error until it is read back, so a failure from a
 * previous run is otherwise reported by the *next* M2CMD -- pointing at the
 * wrong call and the wrong run. */
void clear_error_latch(AWGEngine* pc) {
    char stale[ERRORTEXTLEN] = {0};
    spcm_dwGetErrorInfo_i32(pc->hCard, nullptr, nullptr, stale);
}

bool configure_card(AWGEngine* pc, const AWGEngineConfig* cfg) {
    int32_t fnc_type = 0;
    spcm_dwGetParam_i32(pc->hCard, SPC_FNCTYPE, &fnc_type);
    if (fnc_type != SPCM_TYPE_AO) {
        set_error(pc, "card function type %d is not SPCM_TYPE_AO", fnc_type);
        return false;
    }

    bool ok = check_spc(pc, spcm_dwSetParam_i32(pc->hCard, SPC_CHENABLE, CHANNEL0 | CHANNEL1),
                        "SPC_CHENABLE");

    if (pc->mode == AWG_ENGINE_MODE_MEMORY) {
        ok = ok && check_spc(pc,
                             spcm_dwSetParam_i32(pc->hCard, SPC_CARDMODE, SPC_REP_STD_SEQUENCE),
                             "SPC_CARDMODE(SEQUENCE)");
        ok = ok && check_spc(pc,
                             spcm_dwSetParam_i32(pc->hCard, SPC_SEQMODE_MAXSEGMENTS,
                                                 AWG_SEQ_N_SEGMENTS),
                             "SPC_SEQMODE_MAXSEGMENTS");
        ok = ok && check_spc(pc, spcm_dwSetParam_i32(pc->hCard, SPC_SEQMODE_STARTSTEP, 0),
                             "SPC_SEQMODE_STARTSTEP");
    } else {
        ok = ok && check_spc(pc,
                             spcm_dwSetParam_i32(pc->hCard, SPC_CARDMODE, SPC_REP_FIFO_SINGLE),
                             "SPC_CARDMODE(FIFO)");
        /* SPC_LOOPS = 0 is "replay forever". Without it the card keeps
         * whatever loop count it was left with, streams that much and STOPS
         * -- which presents exactly like an underrun but is deterministic and
         * independent of ring depth. Values from the vendor's own SCAPP
         * example, scapp/cuda_rdma/rdma_fifo_kernel_DA. */
        ok = ok && check_spc(pc, spcm_dwSetParam_i64(pc->hCard, SPC_SEGMENTSIZE, 1024),
                             "SPC_SEGMENTSIZE");
        ok = ok && check_spc(pc, spcm_dwSetParam_i64(pc->hCard, SPC_LOOPS, 0), "SPC_LOOPS");
    }

    ok = ok && check_spc(pc, spcm_dwSetParam_i32(pc->hCard, SPC_TIMEOUT, kStartupTimeoutMs),
                         "SPC_TIMEOUT");
    ok = ok && check_spc(pc,
                         spcm_dwSetParam_i32(pc->hCard, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE),
                         "SPC_TRIG_ORMASK");
    ok = ok && check_spc(pc, spcm_dwSetParam_i32(pc->hCard, SPC_TRIG_ANDMASK, 0),
                         "SPC_TRIG_ANDMASK");
    ok = ok && check_spc(pc, spcm_dwSetParam_i32(pc->hCard, SPC_CLOCKMODE, SPC_CM_INTPLL),
                         "SPC_CLOCKMODE");

    const int32_t amp_mv = (int32_t)(cfg->max_amplitude_v * 1000.0 + 0.5);
    for (int32_t ch = 0; ch < 2 && ok; ++ch) {
        ok = ok && check_spc(pc,
                             spcm_dwSetParam_i32(pc->hCard,
                                                 SPC_ENABLEOUT0 + ch * (SPC_ENABLEOUT1 -
                                                                        SPC_ENABLEOUT0), 1),
                             "SPC_ENABLEOUT");
        ok = ok && check_spc(pc,
                             spcm_dwSetParam_i32(pc->hCard,
                                                 SPC_AMP0 + ch * (SPC_AMP1 - SPC_AMP0), amp_mv),
                             "SPC_AMP");
    }

    /* Not every AO model has selectable termination -- some report "unknown
     * register" for SPC_50OHM. That is a capability difference, not a
     * failure. It matters electrically though: SPC_AMP is specified into
     * 50 ohm, so a high-impedance input (a 1 Mohm scope, an unterminated
     * amplifier) sees roughly TWICE the programmed voltage. */
    if (ok) {
        const int32_t load_50ohm = (cfg->output_load_ohms == 50.0) ? 1 : 0;
        if (spcm_dwSetParam_i32(pc->hCard, SPC_50OHM0, load_50ohm) != ERR_OK) {
            clear_error_latch(pc);
            fprintf(stderr,
                    "[awg_engine] note: no selectable output termination on this card; "
                    "SPC_AMP is into 50 ohm, so an unterminated load sees about 2x %.3f V. "
                    "Terminate externally.\n",
                    cfg->max_amplitude_v);
        } else {
            spcm_dwSetParam_i32(pc->hCard, SPC_50OHM1, load_50ohm);
        }
    }

    /* NB: the driver's own `int64` (dlltyp.h), not `int64_t`. On LP64 Linux
     * `int64_t` is `long` while the SDK's is `long long` -- same width,
     * distinct types, so `int64_t*` will not bind to the out-parameter. */
    if (ok) {
        int64 max_rate = 0;
        spcm_dwGetParam_i64(pc->hCard, SPC_PCISAMPLERATE, &max_rate);
        const int64 target = (cfg->sample_rate_hz > 0.0) ? (int64)cfg->sample_rate_hz : max_rate;
        ok = check_spc(pc, spcm_dwSetParam_i64(pc->hCard, SPC_SAMPLERATE, target),
                       "SPC_SAMPLERATE");
    }
    if (ok) {
        int64 actual_rate = 0;
        spcm_dwGetParam_i64(pc->hCard, SPC_SAMPLERATE, &actual_rate);
        pc->sample_rate_hz = (double)actual_rate;
    }
    /* MEMORY mode: SPC_REP_STD_SEQUENCE's WRITESETUP validates the step
     * table along with everything else, and at this point no segment or
     * step data exists yet -- upload_sequence() (called from load_round())
     * writes that later, and issues WRITESETUP itself once it does. An
     * empty table here reads back as the generic ERR_SETUP ("the setup
     * isn't valid"), same as the vendor's own sequence example: it writes
     * every segment and step before ever validating the setup. STREAM has
     * no step table, so it validates fine right here. */
    if (ok && pc->mode == AWG_ENGINE_MODE_STREAM) {
        ok = check_spc(pc, spcm_dwSetParam_i32(pc->hCard, SPC_M2CMD, M2CMD_CARD_WRITESETUP),
                       "M2CMD_CARD_WRITESETUP");
    }
    if (ok) {
        int32_t max_value = 1;
        spcm_dwGetParam_i32(pc->hCard, SPC_MIINST_MAXADCVALUE, &max_value);
        pc->max_value = (int16_t)max_value;
    }
    return ok;
}

/* MEMORY mode: render the round and its park segment and hand both to the
 * card. Called from load_round(), so this is where the cost lands. */
int upload_sequence(AWGEngine* pc) {
    const int64_t round_frames = awg_seq_align_up(pc->schedule.total_samples);
    if (round_frames > pc->dma_buffer_samples) {
        set_error(pc,
                  "round is %lld samples (%.3f ms) but the staging buffer holds %lld "
                  "(%.3f ms). Raise AWGEngineConfig.dma_buffer_samples (it must still fit "
                  "the GPU's BAR1 aperture: frames x 4 bytes), lower sample_rate_hz, or "
                  "shorten the round.",
                  (long long)round_frames, round_frames / pc->sample_rate_hz * 1e3,
                  (long long)pc->dma_buffer_samples,
                  pc->dma_buffer_samples / pc->sample_rate_hz * 1e3);
        return -1;
    }
    if (pc->hold_tail_samples > pc->dma_buffer_samples) {
        set_error(pc, "hold_tail_samples (%lld) exceeds the staging buffer (%lld)",
                  (long long)pc->hold_tail_samples, (long long)pc->dma_buffer_samples);
        return -1;
    }

    char err[256] = {0};
    awg_schedule_free(&pc->tail);
    if (awg_schedule_hold_tail(&pc->tail, &pc->schedule, pc->schedule.total_samples,
                               pc->hold_tail_samples, err, sizeof(err)) != 0) {
        set_error(pc, "%s", err);
        return -1;
    }

    cudaError_t cerr = awg_device_schedule_upload(&pc->dev_tail, &pc->tail);
    if (!check_cuda(pc, cerr, "tail schedule upload")) {
        return -1;
    }

    clear_error_latch(pc);
    uint32_t e = awg_seq_upload(pc->hCard, pc->dma_buffer, &pc->dev_schedule, 0,
                                AWG_SEQ_SEG_ROUND, round_frames, (float)pc->max_value,
                                pc->stream, &cerr);
    if (!check_cuda(pc, cerr, "render round") || !check_spc(pc, e, "upload round segment")) {
        return -1;
    }
    e = awg_seq_upload(pc->hCard, pc->dma_buffer, &pc->dev_tail, 0, AWG_SEQ_SEG_HOLD,
                       pc->hold_tail_samples, (float)pc->max_value, pc->stream, &cerr);
    if (!check_cuda(pc, cerr, "render park segment") ||
        !check_spc(pc, e, "upload park segment")) {
        return -1;
    }
    if (!check_spc(pc, awg_seq_write_steps(pc->hCard), "SPC_SEQMODE_STEPMEM")) {
        return -1;
    }
    /* Deferred from configure_card(): SPC_REP_STD_SEQUENCE's WRITESETUP
     * validates the step table, which only exists now. */
    if (!check_spc(pc, spcm_dwSetParam_i32(pc->hCard, SPC_M2CMD, M2CMD_CARD_WRITESETUP),
                   "M2CMD_CARD_WRITESETUP")) {
        return -1;
    }

    fprintf(stderr,
            "[awg_engine] uploaded %.3f ms round + %.3f ms looping park to card memory\n",
            round_frames / pc->sample_rate_hz * 1e3,
            pc->hold_tail_samples / pc->sample_rate_hz * 1e3);
    return 0;
}

} // namespace

extern "C" AWGEngine* awg_engine_open(const AWGEngineConfig* cfg) {
    if (cfg == nullptr || cfg->card_path == nullptr) {
        fprintf(stderr, "[awg_engine] open: cfg/card_path is NULL\n");
        return nullptr;
    }
    if (cfg->max_amplitude_v > kMaxSafeOutputV) {
        fprintf(stderr, "[awg_engine] max_amplitude_v=%.3f V exceeds the %.1f V ceiling\n",
                cfg->max_amplitude_v, kMaxSafeOutputV);
        return nullptr;
    }
    if (cfg->grid_rows <= 0 || cfg->grid_cols <= 0) {
        fprintf(stderr, "[awg_engine] grid_rows/grid_cols must be positive\n");
        return nullptr;
    }
    if (cfg->dma_buffer_samples <= 0 ||
        (cfg->mode == AWG_ENGINE_MODE_STREAM &&
         (cfg->notify_samples <= 0 || cfg->dma_buffer_samples % cfg->notify_samples != 0))) {
        fprintf(stderr,
                "[awg_engine] dma_buffer_samples must be positive, and in STREAM mode an "
                "exact multiple of a positive notify_samples (got %lld / %d)\n",
                (long long)cfg->dma_buffer_samples, cfg->notify_samples);
        return nullptr;
    }

    AWGEngine* pc = new AWGEngine();
    pc->mode = cfg->mode;
    pc->notify_samples = cfg->notify_samples;
    pc->dma_buffer_samples = cfg->dma_buffer_samples;
    pc->fill_start_threshold_promille = cfg->fill_start_threshold_promille;
    pc->hold_tail_samples =
        cfg->hold_tail_samples > 0 ? cfg->hold_tail_samples : kDefaultHoldTailSamples;
    pc->cuda_device_index = cfg->cuda_device_index;
    pc->n_tones[0] = cfg->grid_rows;
    pc->n_tones[1] = cfg->grid_cols;

    pc->hCard = spcm_hOpen(const_cast<char*>(cfg->card_path));
    if (pc->hCard == nullptr) {
        fprintf(stderr, "[awg_engine] spcm_hOpen('%s') returned NULL -- no card found\n",
                cfg->card_path);
        delete pc;
        return nullptr;
    }

    bool ok = configure_card(pc, cfg);

    if (ok) {
        const int64_t bytes = pc->dma_buffer_samples * 4;
        pc->dma_buffer = pvGetRDMABuffer(cfg->cuda_device_index, bytes);
        if (pc->dma_buffer == nullptr) {
            /* Almost always BAR1, not free VRAM: the buffer is pinned for
             * GPUDirect RDMA, so it must fit the aperture. Too large and the
             * driver fails in the kernel with "ERROR in BuildSGList:
             * nvidia_p2p_get_pages failed", which says nothing about size. */
            set_error(pc,
                      "pvGetRDMABuffer(%d, %lld bytes = %.0f MB) failed. This buffer is "
                      "pinned for GPUDirect RDMA and must fit the GPU's BAR1 aperture "
                      "(nvidia-smi -q | grep -A3 'BAR1 Memory Usage'; a T1000 has 256 MB). "
                      "Reduce AWGEngineConfig.dma_buffer_samples.",
                      cfg->cuda_device_index, (long long)bytes, bytes / 1e6);
            ok = false;
        }
    }

    /* STREAM defines its transfer once, over the whole ring. MEMORY defines
     * one per sequence segment, in load_round(). */
    if (ok && pc->mode == AWG_ENGINE_MODE_STREAM) {
        ok = check_spc(pc,
                       spcm_dwDefTransfer_i64(pc->hCard, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD,
                                              pc->notify_samples * 4, pc->dma_buffer, 0,
                                              pc->dma_buffer_samples * 4),
                       "spcm_dwDefTransfer_i64");
    }

    ok = ok && check_cuda(pc, cudaSetDevice(cfg->cuda_device_index), "cudaSetDevice");
    ok = ok && check_cuda(pc, cudaStreamCreate(&pc->stream), "cudaStreamCreate");

    if (!ok) {
        if (pc->dma_buffer != nullptr) cudaFree(pc->dma_buffer);
        if (pc->stream != nullptr) cudaStreamDestroy(pc->stream);
        spcm_vClose(pc->hCard);
        delete pc;
        return nullptr;
    }
    return pc;
}

extern "C" double awg_engine_sample_rate_hz(const AWGEngine* pc) {
    return pc ? pc->sample_rate_hz : 0.0;
}

extern "C" int16_t awg_engine_max_sample_value(const AWGEngine* pc) {
    return pc ? pc->max_value : 1;
}

extern "C" int64_t awg_engine_max_round_samples(const AWGEngine* pc) {
    if (pc == nullptr) {
        return 0;
    }
    return pc->mode == AWG_ENGINE_MODE_MEMORY ? pc->dma_buffer_samples : INT64_MAX;
}

extern "C" double awg_engine_total_travel_duration_s(const AWGEngine* pc) {
    if (pc == nullptr || !pc->have_round || pc->sample_rate_hz <= 0.0) {
        return 0.0;
    }
    return (double)pc->schedule.total_samples / pc->sample_rate_hz;
}

extern "C" int awg_engine_load_round(AWGEngine* pc, const double* batch_travel_durations_s,
                                     int32_t n_batches, const AWGRoundRamp* ramps,
                                     int32_t n_ramps, const int32_t* batch_ramp_counts,
                                     int32_t ramp_shape) {
    if (pc == nullptr || batch_travel_durations_s == nullptr || ramps == nullptr ||
        batch_ramp_counts == nullptr || n_batches <= 0) {
        if (pc != nullptr) {
            set_error(pc, "load_round: invalid arguments");
        }
        return -1;
    }
    if (pc->playing || pc->sctx.running.load(std::memory_order_relaxed)) {
        set_error(pc, "load_round called while play() is running -- stop() first");
        return -1;
    }

    char err[256] = {0};
    AwgSchedule sch;
    if (awg_schedule_build(&sch, batch_travel_durations_s, n_batches, ramps, n_ramps,
                           batch_ramp_counts, pc->n_tones[0], pc->n_tones[1],
                           pc->sample_rate_hz, ramp_shape, err, sizeof(err)) != 0) {
        set_error(pc, "%s", err);
        return -1;
    }
    if (sch.total_samples <= 0) {
        awg_schedule_free(&sch);
        set_error(pc, "round has zero total duration (every batch is a hold)");
        return -1;
    }

    pc->have_round = false;
    awg_schedule_free(&pc->schedule);
    pc->schedule = sch;

    if (!check_cuda(pc, awg_device_schedule_upload(&pc->dev_schedule, &pc->schedule),
                    "schedule upload")) {
        awg_schedule_free(&pc->schedule);
        return -1;
    }
    if (pc->mode == AWG_ENGINE_MODE_MEMORY && upload_sequence(pc) != 0) {
        awg_schedule_free(&pc->schedule);
        return -1;
    }

    pc->have_round = true;
    return 0;
}

extern "C" int awg_engine_play(AWGEngine* pc) {
    if (pc == nullptr) {
        return -1;
    }
    if (!pc->have_round) {
        set_error(pc, "play called before a successful load_round()");
        return -1;
    }
    clear_error_latch(pc);

    if (pc->mode == AWG_ENGINE_MODE_MEMORY) {
        /* Everything is already on the card; just trigger it. The sequence
         * plays the round once and then loops the park segment forever. */
        if (!check_spc(pc,
                       spcm_dwSetParam_i32(pc->hCard, SPC_M2CMD,
                                           M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER),
                       "M2CMD_CARD_START")) {
            return -1;
        }
        pc->playing = true;
        return 0;
    }

    if (pc->sctx.running.load(std::memory_order_relaxed)) {
        set_error(pc, "play called while already running");
        return -1;
    }
    if (!check_spc(pc, spcm_dwSetParam_i32(pc->hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA),
                   "M2CMD_DATA_STARTDMA")) {
        return -1;
    }

    AwgStreamCtx* c = &pc->sctx;
    c->hCard = pc->hCard;
    c->ring = pc->dma_buffer;
    c->ring_frames = pc->dma_buffer_samples;
    c->notify_frames = pc->notify_samples;
    c->fill_start_threshold_promille = pc->fill_start_threshold_promille;
    c->max_render_frames = 1 << 20;
    c->cuda_device_index = pc->cuda_device_index;
    c->max_value = (float)pc->max_value;
    c->sched = &pc->dev_schedule;
    c->stream = pc->stream;
    c->stop_flag.store(false, std::memory_order_relaxed);
    c->running.store(true, std::memory_order_relaxed);
    c->card_started = false;
    c->start_failed = false;
    c->failed = 0;
    c->err[0] = '\0';

    pc->pump_thread = std::thread(awg_stream_pump, c);

    std::unique_lock<std::mutex> lock(c->start_mutex);
    const bool reached = c->start_cv.wait_for(lock, std::chrono::milliseconds(kStartupTimeoutMs),
                                              [c] { return c->card_started; });
    const bool failed = reached && c->start_failed;
    lock.unlock();

    if (!reached) {
        set_error(pc, "pump did not reach the fill threshold within %d ms", kStartupTimeoutMs);
    } else if (failed && c->failed) {
        set_error(pc, "%s", c->err);
    }
    if (!reached || failed) {
        awg_engine_stop(pc);
        return -1;
    }
    return 0;
}

extern "C" const char* awg_engine_last_error(const AWGEngine* pc) {
    if (pc == nullptr) {
        return "engine handle is NULL";
    }
    return pc->last_error[0] != '\0' ? pc->last_error : nullptr;
}

extern "C" void awg_engine_stop(AWGEngine* pc) {
    if (pc == nullptr) {
        return;
    }
    pc->playing = false;
    pc->sctx.stop_flag.store(true, std::memory_order_relaxed);
    if (pc->hCard != nullptr) {
        spcm_dwSetParam_i32(pc->hCard, SPC_M2CMD, M2CMD_DATA_STOPDMA | M2CMD_CARD_STOP);
    }
    if (pc->pump_thread.joinable()) {
        pc->pump_thread.join();
    }
    pc->sctx.running.store(false, std::memory_order_relaxed);
    if (pc->sctx.failed && pc->last_error[0] == '\0') {
        set_error(pc, "%s", pc->sctx.err);
    }
}

extern "C" void awg_engine_close(AWGEngine* pc) {
    if (pc == nullptr) {
        return;
    }
    awg_engine_stop(pc);
    awg_device_schedule_free(&pc->dev_schedule);
    awg_device_schedule_free(&pc->dev_tail);
    awg_schedule_free(&pc->schedule);
    awg_schedule_free(&pc->tail);
    if (pc->stream != nullptr) {
        cudaStreamDestroy(pc->stream);
    }
    if (pc->dma_buffer != nullptr) {
        cudaFree(pc->dma_buffer);
    }
    if (pc->hCard != nullptr) {
        spcm_vClose(pc->hCard);
    }
    delete pc;
}
