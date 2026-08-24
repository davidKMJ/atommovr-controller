/* Round -> segment schedule.
 *
 * Flattens a round into a table of AwgSegment with absolute start samples and
 * the phase carry already resolved. The carry is sequential but trivial (500
 * batches x 60 tones is 30k segments, ~2 MB, microseconds), and doing it up
 * front is what lets the GPU render an arbitrary span in a *single* launch.
 * That matters at the experiment scale: a 5 us batch is about one kernel
 * launch, so launch-per-batch rendering could never be real-time.
 *
 * Header-only and CUDA-free, so it unit-tests on a laptop against the Python
 * reference in awg_controller.scapp.
 */

#ifndef AWG_ENGINE_SCHEDULE_H
#define AWG_ENGINE_SCHEDULE_H

#include "awg_engine.h"
#include "phase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct AwgSchedule {
    AwgSegment* segments;   /* n_batches * n_tones_total, batch-major */
    int64_t* batch_start;   /* n_batches + 1 absolute sample boundaries */
    int32_t n_batches;
    int32_t n_tones[2];     /* per channel */
    int32_t n_tones_total;  /* n_tones[0] + n_tones[1] */
    int64_t total_samples;  /* unpadded round length */
    double sample_rate_hz;
} AwgSchedule;

/* Global tone index for (channel, tone_index). */
AWG_HD inline int32_t awg_tone_slot(const AwgSchedule* sch, int32_t channel,
                                    int32_t tone_index) {
    return (channel == 0) ? tone_index : sch->n_tones[0] + tone_index;
}

/* Batch covering `abs_sample` given raw batch-start boundaries, clamped to the
 * last batch so that sampling past the end of the round keeps holding the
 * final state -- the indefinite park falls out of the same lookup rather than
 * being a special case. Takes raw arrays (not AwgSchedule) so the identical
 * search can run against both the host AwgSchedule and the device-resident
 * AwgDeviceSchedule (render.cuh) from a single definition. */
AWG_HD inline int32_t awg_batch_at(const int64_t* batch_start, int32_t n_batches,
                                   int64_t total_samples, int64_t abs_sample) {
    if (abs_sample <= 0) {
        return 0;
    }
    if (abs_sample >= total_samples) {
        return n_batches - 1;
    }
    int32_t lo = 0, hi = n_batches - 1;
    while (lo < hi) {
        const int32_t mid = (lo + hi + 1) >> 1;
        if (batch_start[mid] <= abs_sample) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

AWG_HD inline int32_t awg_schedule_batch_at(const AwgSchedule* sch, int64_t abs_sample) {
    return awg_batch_at(sch->batch_start, sch->n_batches, sch->total_samples, abs_sample);
}

AWG_HD inline const AwgSegment* awg_schedule_segment(const AwgSchedule* sch, int32_t batch,
                                                     int32_t slot) {
    return &sch->segments[(int64_t)batch * sch->n_tones_total + slot];
}

static inline void awg_schedule_free(AwgSchedule* sch) {
    if (sch == NULL) {
        return;
    }
    free(sch->segments);
    free(sch->batch_start);
    sch->segments = NULL;
    sch->batch_start = NULL;
    sch->n_batches = 0;
    sch->total_samples = 0;
}

/* Build the schedule. Returns 0 on success, -1 with a message in `err`.
 *
 * Every batch must supply exactly one ramp per tone (n_tones[0]+n_tones[1]),
 * matching RFConverter's full-grid-every-batch invariant. A batch whose
 * duration is <= 0 contributes no samples but still advances tone state.
 */
static inline int awg_schedule_build(AwgSchedule* sch, const double* batch_travel_durations_s,
                                     int32_t n_batches, const AWGRoundRamp* ramps,
                                     int32_t n_ramps, const int32_t* batch_ramp_counts,
                                     int32_t n_tones_v, int32_t n_tones_h,
                                     double sample_rate_hz, int32_t ramp_shape, char* err,
                                     size_t errlen) {
    memset(sch, 0, sizeof(*sch));
    if (n_batches <= 0 || sample_rate_hz <= 0.0) {
        snprintf(err, errlen, "schedule: n_batches=%d, sample_rate=%g", n_batches,
                 sample_rate_hz);
        return -1;
    }
    const int32_t total_tones = n_tones_v + n_tones_h;
    int64_t want = 0;
    for (int32_t b = 0; b < n_batches; ++b) {
        if (batch_ramp_counts[b] != total_tones) {
            snprintf(err, errlen, "schedule: batch %d has %d ramps, expected %d", b,
                     batch_ramp_counts[b], total_tones);
            return -1;
        }
        want += batch_ramp_counts[b];
    }
    if (want != (int64_t)n_ramps) {
        snprintf(err, errlen, "schedule: ramp counts sum to %lld, got n_ramps=%d",
                 (long long)want, n_ramps);
        return -1;
    }

    sch->n_batches = n_batches;
    sch->n_tones[0] = n_tones_v;
    sch->n_tones[1] = n_tones_h;
    sch->n_tones_total = total_tones;
    sch->sample_rate_hz = sample_rate_hz;
    sch->segments =
        (AwgSegment*)calloc((size_t)n_batches * (size_t)total_tones, sizeof(AwgSegment));
    sch->batch_start = (int64_t*)calloc((size_t)n_batches + 1, sizeof(int64_t));
    AwgSegment* prev = (AwgSegment*)calloc((size_t)total_tones, sizeof(AwgSegment));
    if (sch->segments == NULL || sch->batch_start == NULL || prev == NULL) {
        free(prev);
        awg_schedule_free(sch);
        snprintf(err, errlen, "schedule: out of memory");
        return -1;
    }

    /* Pre-round resting state: each tone parked at the frequency the first
     * batch names as its f_start, at zero phase. Mirrors the Python path. */
    for (int32_t r = 0; r < batch_ramp_counts[0]; ++r) {
        const AWGRoundRamp* rp = &ramps[r];
        const int32_t slot = awg_tone_slot(sch, rp->channel, rp->tone_index);
        awg_segment_build(&prev[slot], rp->f_start_hz, rp->f_start_hz, 0.0, 0, sample_rate_hz,
                          0, 0ull, 0.0f);
    }

    int64_t cursor = 0;
    int32_t off = 0;
    for (int32_t b = 0; b < n_batches; ++b) {
        sch->batch_start[b] = cursor;
        const double dur = batch_travel_durations_s[b];
        const int is_scurve = (dur > 0.0) && (ramp_shape == AWG_ENGINE_SHAPE_SCURVE);

        for (int32_t r = 0; r < batch_ramp_counts[b]; ++r) {
            const AWGRoundRamp* rp = &ramps[off + r];
            if (rp->channel < 0 || rp->channel > 1) {
                free(prev);
                awg_schedule_free(sch);
                snprintf(err, errlen, "schedule: batch %d ramp %d has channel %d", b, r,
                         rp->channel);
                return -1;
            }
            if (rp->tone_index < 0 || rp->tone_index >= sch->n_tones[rp->channel]) {
                free(prev);
                awg_schedule_free(sch);
                snprintf(err, errlen, "schedule: batch %d tone_index %d out of range for ch %d",
                         b, rp->tone_index, rp->channel);
                return -1;
            }
            const int32_t slot = awg_tone_slot(sch, rp->channel, rp->tone_index);
            /* dynamic phase only -- see AwgSegment::static_phase_q64 */
            const uint64_t carry = awg_segment_phase_q64(&prev[slot], cursor);

            AwgSegment* seg = &sch->segments[(int64_t)b * total_tones + slot];
            awg_segment_build(seg, rp->f_start_hz, rp->f_end_hz, dur, is_scurve,
                              sample_rate_hz, cursor, carry,
                              (float)(rp->amplitude_pct / 100.0));
            double turns = rp->phase_deg / 360.0;
            turns -= floor(turns);
            seg->static_phase_q64 = (uint64_t)(turns * AWG_Q64_SCALE);
            prev[slot] = *seg;
        }

        off += batch_ramp_counts[b];
        if (dur > 0.0) {
            cursor += (int64_t)(dur * sample_rate_hz + 0.5);
        }
    }
    sch->batch_start[n_batches] = cursor;
    sch->total_samples = cursor;

    free(prev);
    return 0;
}

/* Build a one-batch schedule that parks every tone where `round` left it at
 * `at_sample`, for MEMORY mode's looping tail segment.
 *
 * The card replays this segment end-to-end forever, so it must contain a
 * whole number of cycles of every tone or each wrap injects a phase step.
 * Two things make that exact:
 *   - tail_samples is a power of two, so a Q0.64 rate times it wraps cleanly;
 *   - each frequency is snapped to the nearest multiple of
 *     sample_rate/tail_samples.
 * At 1.25 GS/s with a 2^20 tail that grid is ~1.2 kHz, i.e. ~15 ppm on an
 * 80 MHz tone -- orders of magnitude below AOD pointing resolution.
 *
 * Phase carries exactly from the round, so the round->tail seam is clean too.
 */
static inline int awg_schedule_hold_tail(AwgSchedule* tail, const AwgSchedule* round,
                                         int64_t at_sample, int64_t tail_samples, char* err,
                                         size_t errlen) {
    memset(tail, 0, sizeof(*tail));
    if (tail_samples <= 0 || (tail_samples & (tail_samples - 1)) != 0) {
        snprintf(err, errlen, "hold tail: %lld samples is not a power of two",
                 (long long)tail_samples);
        return -1;
    }

    const int32_t n = round->n_tones_total;
    tail->n_batches = 1;
    tail->n_tones[0] = round->n_tones[0];
    tail->n_tones[1] = round->n_tones[1];
    tail->n_tones_total = n;
    tail->sample_rate_hz = round->sample_rate_hz;
    tail->total_samples = tail_samples;
    tail->segments = (AwgSegment*)calloc((size_t)n, sizeof(AwgSegment));
    tail->batch_start = (int64_t*)calloc(2, sizeof(int64_t));
    if (tail->segments == NULL || tail->batch_start == NULL) {
        awg_schedule_free(tail);
        snprintf(err, errlen, "hold tail: out of memory");
        return -1;
    }
    tail->batch_start[1] = tail_samples;

    const AwgSegment* last = &round->segments[(int64_t)(round->n_batches - 1) * n];
    for (int32_t slot = 0; slot < n; ++slot) {
        const AwgSegment* src = &last[slot];
        /* a_end_q64 is f_end/fs in Q0.64; f_end < fs/2 so it never wrapped. */
        const double f_end = (double)src->a_end_q64 / AWG_Q64_SCALE * round->sample_rate_hz;
        const double grid = round->sample_rate_hz / (double)tail_samples;
        const double f_hold = floor(f_end / grid + 0.5) * grid;

        awg_segment_build(&tail->segments[slot], f_hold, f_hold, 0.0, 0,
                          round->sample_rate_hz, 0,
                          awg_segment_phase_q64(src, at_sample), src->amplitude);
        tail->segments[slot].static_phase_q64 = src->static_phase_q64;
    }
    return 0;
}

#endif /* AWG_ENGINE_SCHEDULE_H */
