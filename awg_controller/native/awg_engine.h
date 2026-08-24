/*
 * awg_engine.h -- two-mode SCAPP replay engine for a Spectrum M4i AO card.
 *
 * A round is a list of batches; each batch names a frequency ramp for *every*
 * tone (RFConverter's full-grid invariant) and a travel duration. The engine
 * resolves that into a phase-continuous two-channel waveform and plays it.
 *
 * The two modes differ only in how samples reach the card:
 *
 *   STREAM  FIFO replay. Samples are rendered just ahead of the card's read
 *           pointer, straight into the SCAPP RDMA ring. Round length is
 *           unbounded (memory is the ring alone), but the PCIe link must
 *           sustain sample_rate*4 B/s forever -- an M4i is Gen2 x8, so this
 *           caps out near 500-800 MS/s two-channel. Use for long, slow,
 *           watch-it-on-a-scope rounds.
 *
 *   MEMORY  Sequence replay from the card's own DRAM. The whole round is
 *           rendered up front, uploaded once, and played by the card with no
 *           sustained streaming -- so it runs at the full 1.25 GS/s. Round
 *           length is bounded by dma_buffer_samples (see below). Use for
 *           real experiment rounds, which are milliseconds long.
 *
 * In both modes the engine parks on the round's final frequencies
 * indefinitely and phase-exactly once the round is exhausted.
 */

#ifndef AWG_CONTROLLER_AWG_ENGINE_H
#define AWG_CONTROLLER_AWG_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    AWG_ENGINE_MODE_STREAM = 0,
    AWG_ENGINE_MODE_MEMORY = 1,
};

/* Frequency-ramp shape for batches with travel_duration_s > 0. */
enum {
    AWG_ENGINE_SHAPE_LINEAR = 0,
    AWG_ENGINE_SHAPE_SCURVE = 1,
};

typedef struct {
    const char* card_path;
    double max_amplitude_v;
    /* Best-effort. M4i.66xx AWGs have a fixed 50 ohm output and do not
     * implement SPC_50OHM (a digitizer *input* register), so this is ignored
     * there. max_amplitude_v is always the amplitude into 50 ohm; an
     * unterminated load sees roughly twice it. */
    double output_load_ohms;
    int32_t mode;               /* AWG_ENGINE_MODE_* */
    int32_t notify_samples;     /* STREAM only: render budget per wake */
    /* STREAM: ring depth. MEMORY: staging-buffer size, hence the maximum
     * round length. Either way this is pinned for GPUDirect RDMA and so must
     * fit the GPU's BAR1 aperture (a T1000 has 256 MB; frames are 4 bytes). */
    int64_t dma_buffer_samples;
    int32_t fill_start_threshold_promille;  /* STREAM only */
    /* MEMORY only: length of the looped park segment. Must be a power of two
     * so the segment holds a whole number of cycles of every tone. 0 -> 2^20. */
    int64_t hold_tail_samples;
    double sample_rate_hz;      /* 0 => the card's maximum */
    int32_t grid_rows;          /* tone count, channel 0 (V/row AOD) */
    int32_t grid_cols;          /* tone count, channel 1 (H/col AOD) */
    int32_t cuda_device_index;
} AWGEngineConfig;

typedef struct {
    int32_t channel;      /* 0 or 1 */
    int32_t tone_index;   /* 0..grid_rows-1 (ch0) or 0..grid_cols-1 (ch1) */
    double f_start_hz;
    double f_end_hz;
    double amplitude_pct; /* 0-100 */
    double phase_deg;
} AWGRoundRamp;

typedef struct AWGEngine AWGEngine;

/* Opens the card and CUDA device and negotiates the sample rate. Returns NULL
 * on failure; detail goes to stderr since no handle exists yet to hold it. */
AWGEngine* awg_engine_open(const AWGEngineConfig* cfg);

double awg_engine_sample_rate_hz(const AWGEngine* pc);
int16_t awg_engine_max_sample_value(const AWGEngine* pc);

/* Longest round (in samples) this engine can accept. INT64_MAX in STREAM
 * mode; dma_buffer_samples in MEMORY mode. */
int64_t awg_engine_max_round_samples(const AWGEngine* pc);

/* Resolves a round into a segment schedule, replacing any previous round.
 *
 * `ramps` is every batch's ramps concatenated in batch order;
 * `batch_ramp_counts[b]` says how many belong to batch b (each must equal
 * grid_rows + grid_cols); `batch_travel_durations_s[b]` is its travel window,
 * where <= 0 means a hold that contributes no samples but still advances tone
 * state. The first batch's f_start per tone is the pre-existing resting
 * frequency, matching awg_controller.scapp.synthesize_round_waveform.
 *
 * In MEMORY mode this also renders and uploads the waveform, so it is the
 * expensive call; in STREAM mode only the schedule is built (~2 MB for 500
 * batches x 60 tones) and samples are rendered during play().
 *
 * Returns 0 on success. Must precede play(); safe to call again once stopped. */
int awg_engine_load_round(AWGEngine* pc, const double* batch_travel_durations_s,
                          int32_t n_batches, const AWGRoundRamp* ramps, int32_t n_ramps,
                          const int32_t* batch_ramp_counts, int32_t ramp_shape);

/* Summed travel window (s) over the loaded round. A `total_` name because it
 * is a sum; a single batch's window is a `travel_duration_s`. */
double awg_engine_total_travel_duration_s(const AWGEngine* pc);

/* Starts playback. STREAM blocks until the ring has pre-filled past
 * fill_start_threshold_promille and the card has started; MEMORY returns as
 * soon as the card is triggered. Returns 0 on success. */
int awg_engine_play(AWGEngine* pc);

/* NULL if the engine has not failed. Owned by `pc`, valid until the next
 * awg_engine_* call on it. */
const char* awg_engine_last_error(const AWGEngine* pc);

/* Stops playback. Safe to call repeatedly and before play(). */
void awg_engine_stop(AWGEngine* pc);

/* Stops, frees GPU buffers, closes the card, frees `pc`. */
void awg_engine_close(AWGEngine* pc);

#ifdef __cplusplus
}
#endif

#endif /* AWG_CONTROLLER_AWG_ENGINE_H */
