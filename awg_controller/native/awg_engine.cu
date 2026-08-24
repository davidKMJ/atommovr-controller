/* Author: Claude Code, David Ko
 * Facade over the two replay modes (see awg_engine.h).
 *
 * Card setup, error handling and lifetime live here; the modes themselves are
 * stream.cuh (FIFO render-ahead) and sequence.cuh (precompute + card DRAM).
 * Phase maths is fixed point (phase.h), shared verbatim between the host-side
 * carry and the device kernel; see `make test`.
 */

#include "sequence.cuh"
#include "stream.cuh"

#include "../../scapp/cuda_rdma/common/spcm_cuda_common.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <thread>

namespace {
constexpr double kMaxSafeOutputV = 2.0;
constexpr int kStartupTimeoutMs = 5000;
constexpr int64_t kDefaultHoldTailSamples = 1 << 20;
constexpr size_t kErrorCap = 512;
} // namespace

struct AWGEngine {
    drv_handle hCard = nullptr;
    void* dma_buffer = nullptr; /* STREAM: the ring. MEMORY: upload staging. */

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

    AwgDeviceSchedule dev_schedule{}; /* STREAM: live. MEMORY: dropped after upload. */
    int64_t total_samples = 0;

    AwgStreamCtx sctx{}; /* STREAM only */
    std::thread pump_thread;
    bool playing = false;

    std::mutex error_mutex;
    char last_error[kErrorCap] = {0};
};

namespace {

/* Errors are latched, not printed: the Python wrapper raises with the
 * string, and C callers read awg_engine_last_error. Info/warn go to stderr
 * with a tagged prefix. STREAM pump failures after the card has started
 * print once from stream.cuh (play() has already returned). */
char g_last_error[kErrorCap] = {0};
std::mutex g_error_mutex;

void set_error(AWGEngine* pc, const char* fmt, ...) {
    char buf[kErrorCap];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (pc != nullptr) {
        std::lock_guard<std::mutex> lock(pc->error_mutex);
        snprintf(pc->last_error, sizeof(pc->last_error), "%s", buf);
    }
    std::lock_guard<std::mutex> lock(g_error_mutex);
    snprintf(g_last_error, sizeof(g_last_error), "%s", buf);
}

void clear_error(AWGEngine* pc) {
    if (pc != nullptr) {
        std::lock_guard<std::mutex> lock(pc->error_mutex);
        pc->last_error[0] = '\0';
    }
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error[0] = '\0';
}

void log_msg(const char* level, const char* fmt, ...) {
    fprintf(stderr, "[awg_engine] %s: ", level);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
}

bool check_spc(AWGEngine* pc, uint32_t dwErr, const char* what) {
    if (dwErr == ERR_OK) {
        return true;
    }
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

    if (ok) {
        const int32_t load_50ohm = (cfg->output_load_ohms == 50.0) ? 1 : 0;
        if (spcm_dwSetParam_i32(pc->hCard, SPC_50OHM0, load_50ohm) != ERR_OK) {
            clear_error_latch(pc);
            log_msg("warn",
                    "no selectable output termination on this card; "
                    "SPC_AMP is into 50 ohm, so an unterminated load sees about 2x %.3f V. "
                    "Terminate externally.",
                    cfg->max_amplitude_v);
        } else {
            spcm_dwSetParam_i32(pc->hCard, SPC_50OHM1, load_50ohm);
        }
    }

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
 * card. */
int upload_sequence(AWGEngine* pc, const AwgSchedule* round) {
    AwgSchedule tail{};
    AwgDeviceSchedule dev_tail{};
    auto drop = [&] {
        awg_schedule_free(&tail);
        awg_device_schedule_free(&dev_tail);
        awg_device_schedule_free(&pc->dev_schedule);
    };

    const int64_t round_frames = awg_seq_align_up(round->total_samples);
    if (round_frames > pc->dma_buffer_samples) {
        set_error(pc,
                  "round is %lld samples (%.3f ms) but the staging buffer holds %lld "
                  "(%.3f ms). Raise AWGEngineConfig.dma_buffer_samples (it must still fit "
                  "the GPU's BAR1 aperture: frames x 4 bytes), lower sample_rate_hz, or "
                  "shorten the round.",
                  (long long)round_frames, round_frames / pc->sample_rate_hz * 1e3,
                  (long long)pc->dma_buffer_samples,
                  pc->dma_buffer_samples / pc->sample_rate_hz * 1e3);
        drop();
        return -1;
    }
    if (pc->hold_tail_samples > pc->dma_buffer_samples) {
        set_error(pc, "hold_tail_samples (%lld) exceeds the staging buffer (%lld)",
                  (long long)pc->hold_tail_samples, (long long)pc->dma_buffer_samples);
        drop();
        return -1;
    }

    char err[256] = {0};
    if (awg_schedule_hold_tail(&tail, round, round->total_samples, pc->hold_tail_samples, err,
                               sizeof(err)) != 0) {
        set_error(pc, "%s", err);
        drop();
        return -1;
    }
    if (!check_cuda(pc, awg_device_schedule_upload(&dev_tail, &tail), "tail schedule upload")) {
        drop();
        return -1;
    }

    clear_error_latch(pc);
    cudaError_t cerr;
    uint32_t e = awg_seq_upload(pc->hCard, pc->dma_buffer, &pc->dev_schedule, 0,
                                AWG_SEQ_SEG_ROUND, round_frames, (float)pc->max_value,
                                pc->stream, &cerr);
    if (!check_cuda(pc, cerr, "render round") || !check_spc(pc, e, "upload round segment")) {
        drop();
        return -1;
    }
    e = awg_seq_upload(pc->hCard, pc->dma_buffer, &dev_tail, 0, AWG_SEQ_SEG_HOLD,
                       pc->hold_tail_samples, (float)pc->max_value, pc->stream, &cerr);
    if (!check_cuda(pc, cerr, "render park segment") ||
        !check_spc(pc, e, "upload park segment")) {
        drop();
        return -1;
    }
    if (!check_spc(pc, awg_seq_write_steps(pc->hCard), "SPC_SEQMODE_STEPMEM") ||
        !check_spc(pc, spcm_dwSetParam_i32(pc->hCard, SPC_M2CMD, M2CMD_CARD_WRITESETUP),
                   "M2CMD_CARD_WRITESETUP")) {
        drop();
        return -1;
    }

    log_msg("info", "uploaded %.3f ms round + %.3f ms looping park to card memory",
            round_frames / pc->sample_rate_hz * 1e3,
            pc->hold_tail_samples / pc->sample_rate_hz * 1e3);
    drop();
    return 0;
}

} // namespace

extern "C" AWGEngine* awg_engine_open(const AWGEngineConfig* cfg) {
    if (cfg == nullptr || cfg->card_path == nullptr) {
        set_error(nullptr, "open: cfg/card_path is NULL");
        return nullptr;
    }
    if (cfg->max_amplitude_v > kMaxSafeOutputV) {
        set_error(nullptr, "max_amplitude_v=%.3f V exceeds the %.1f V ceiling",
                  cfg->max_amplitude_v, kMaxSafeOutputV);
        return nullptr;
    }
    if (cfg->grid_rows <= 0 || cfg->grid_cols <= 0) {
        set_error(nullptr, "grid_rows/grid_cols must be positive");
        return nullptr;
    }
    if (cfg->dma_buffer_samples <= 0 ||
        (cfg->mode == AWG_ENGINE_MODE_STREAM &&
         (cfg->notify_samples <= 0 || cfg->dma_buffer_samples % cfg->notify_samples != 0))) {
        set_error(nullptr,
                  "dma_buffer_samples must be positive, and in STREAM mode an "
                  "exact multiple of a positive notify_samples (got %lld / %d)",
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
        set_error(pc, "spcm_hOpen('%s') returned NULL -- no card found", cfg->card_path);
        awg_engine_close(pc);
        return nullptr;
    }

    bool ok = configure_card(pc, cfg);

    if (ok) {
        const int64_t bytes = pc->dma_buffer_samples * 4;
        pc->dma_buffer = pvGetRDMABuffer(cfg->cuda_device_index, bytes);
        if (pc->dma_buffer == nullptr) {
            set_error(pc,
                      "pvGetRDMABuffer(%d, %lld bytes = %.0f MB) failed. This buffer is "
                      "pinned for GPUDirect RDMA and must fit the GPU's BAR1 aperture "
                      "(nvidia-smi -q | grep -A3 'BAR1 Memory Usage'). "
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
        awg_engine_close(pc);
        return nullptr;
    }
    clear_error(pc);
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
    if (pc == nullptr || pc->total_samples <= 0 || pc->sample_rate_hz <= 0.0) {
        return 0.0;
    }
    return (double)pc->total_samples / pc->sample_rate_hz;
}

extern "C" int awg_engine_load_round(AWGEngine* pc, const double* batch_travel_durations_s,
                                     int32_t n_batches, const AWGRoundRamp* ramps,
                                     int32_t n_ramps, const int32_t* batch_ramp_counts,
                                     int32_t ramp_shape) {
    if (pc == nullptr || batch_travel_durations_s == nullptr || ramps == nullptr ||
        batch_ramp_counts == nullptr || n_batches <= 0) {
        set_error(pc, "load_round: invalid arguments");
        return -1;
    }
    if (pc->playing) {
        set_error(pc, "load_round called while play() is running -- stop() first");
        return -1;
    }

    char err[256] = {0};
    AwgSchedule sch{};
    if (awg_schedule_build(&sch, batch_travel_durations_s, n_batches, ramps, n_ramps,
                           batch_ramp_counts, pc->n_tones[0], pc->n_tones[1],
                           pc->sample_rate_hz, ramp_shape, err, sizeof(err)) != 0) {
        set_error(pc, "%s", err);
        return -1;
    }
    const int64_t n = sch.total_samples;
    if (n <= 0) {
        awg_schedule_free(&sch);
        set_error(pc, "round has zero total duration (every batch is a hold)");
        return -1;
    }

    pc->total_samples = 0;
    if (!check_cuda(pc, awg_device_schedule_upload(&pc->dev_schedule, &sch),
                    "schedule upload")) {
        awg_schedule_free(&sch);
        return -1;
    }
    if (pc->mode == AWG_ENGINE_MODE_MEMORY && upload_sequence(pc, &sch) != 0) {
        awg_schedule_free(&sch);
        return -1;
    }
    awg_schedule_free(&sch);

    pc->total_samples = n;
    clear_error(pc);
    return 0;
}

extern "C" int awg_engine_play(AWGEngine* pc) {
    if (pc == nullptr) {
        set_error(nullptr, "play called on NULL engine");
        return -1;
    }
    if (pc->total_samples <= 0) {
        set_error(pc, "play called before a successful load_round()");
        return -1;
    }
    if (pc->playing) {
        set_error(pc, "play called while already running");
        return -1;
    }
    clear_error_latch(pc);

    if (pc->mode == AWG_ENGINE_MODE_MEMORY) {
        if (!check_spc(pc,
                       spcm_dwSetParam_i32(pc->hCard, SPC_M2CMD,
                                           M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER),
                       "M2CMD_CARD_START")) {
            return -1;
        }
        pc->playing = true;
        clear_error(pc);
        return 0;
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
    c->card_started = false;
    c->start_failed = false;
    c->error_mutex = &pc->error_mutex;
    c->last_error = pc->last_error;
    c->last_error_len = sizeof(pc->last_error);
    c->failed = 0;
    c->err[0] = '\0';
    clear_error(pc);

    pc->playing = true;
    pc->pump_thread = std::thread(awg_stream_pump, c);

    std::unique_lock<std::mutex> lock(c->start_mutex);
    const bool reached = c->start_cv.wait_for(lock, std::chrono::milliseconds(kStartupTimeoutMs),
                                              [c] { return c->card_started; });
    const bool failed = reached && c->start_failed;
    lock.unlock();

    if (!reached) {
        awg_engine_stop(pc);
        set_error(pc, "pump did not reach the fill threshold within %d ms", kStartupTimeoutMs);
        return -1;
    }
    return failed ? awg_engine_stop(pc) : 0;
}

extern "C" const char* awg_engine_last_error(const AWGEngine* pc) {
    if (pc == nullptr) {
        return g_last_error[0] != '\0' ? g_last_error : nullptr;
    }
    return pc->last_error[0] != '\0' ? pc->last_error : nullptr;
}

extern "C" int awg_engine_stop(AWGEngine* pc) {
    if (pc == nullptr) {
        set_error(nullptr, "stop called on NULL engine");
        return -1;
    }
    pc->sctx.stop_flag.store(true, std::memory_order_relaxed);
    if (pc->hCard != nullptr) {
        spcm_dwSetParam_i32(pc->hCard, SPC_M2CMD, M2CMD_DATA_STOPDMA | M2CMD_CARD_STOP);
    }
    if (pc->pump_thread.joinable()) {
        pc->pump_thread.join();
    }
    pc->playing = false;
    if (!pc->sctx.failed) {
        return 0;
    }
    set_error(pc, "%s", pc->sctx.err[0] != '\0' ? pc->sctx.err : "STREAM pump failed");
    pc->sctx.failed = 0;
    return -1;
}

extern "C" void awg_engine_close(AWGEngine* pc) {
    if (pc == nullptr) {
        return;
    }
    awg_engine_stop(pc);
    awg_device_schedule_free(&pc->dev_schedule);
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
