/* Fixed-point tone phase.
 *
 * Phase is carried in cycles as Q0.64 (uint64). Unsigned wrap at 2^64 is
 * exactly one cycle, so reduction mod 2*pi is free and magnitude-independent.
 */

#ifndef AWG_ENGINE_PHASE_H
#define AWG_ENGINE_PHASE_H

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define AWG_HD __host__ __device__
#else
#define AWG_HD
#endif

#ifndef AWG_PI
#define AWG_PI 3.14159265358979323846
#endif

#define AWG_Q32_SCALE 4294967296.0            /* 2^32 */
#define AWG_Q64_SCALE 18446744073709551616.0  /* 2^64 */

typedef struct AwgSegment {
    int64_t start_sample;   /* absolute sample index where this segment begins */
    int64_t n_ramp;         /* ramp length in samples; <=0 means a pure tone    */

    uint64_t phase0_q64;    /* phase at start_sample                            */
    uint64_t phase_end_q64; /* phase at start_sample + n_ramp                   */

    uint64_t a_q64;         /* cycles/sample during the ramp                    */
    uint64_t a_end_q64;     /* cycles/sample after the ramp (f_end / fs)        */
    /* HALF the chirp rate, i.e. (b/2) in Q0.64, where b is cycles/sample^2.
     * The factor of 2 is folded in so the chirp term is a plain wrapping
     * multiply (b_half * i * i) with no shift. That matters for descending
     * ramps: a negative b is stored two's-complement, and uint64 multiply
     * wraps correctly for it, whereas a 128-bit unsigned product followed
     * by >>1 silently produces garbage. 0 for s-curve. */
    uint64_t b_half_q64;

    /* Per-ramp static offset (phase_deg). Deliberately NOT folded into
     * phase0_q64: the batch-to-batch carry must use the outgoing segment's
     * dynamic phase only, since each incoming segment re-adds its own
     * static offset. Folding it into the carry double-counts a nonzero
     * phase_deg. Added back only at render time. */
    uint64_t static_phase_q64;

    float scurve_cyc;       /* s-curve deviation amplitude in cycles; 0=linear  */
    float amplitude;        /* DAC amplitude scale for this tone                */
} AwgSegment;

/* Phase (cycles, Q0.64) of `seg` at absolute sample `abs_sample`.
 *
 * Past the ramp the tone continues at f_end forever
 */
AWG_HD inline uint64_t awg_segment_phase_q64(const AwgSegment* seg, int64_t abs_sample) {
    int64_t i = abs_sample - seg->start_sample;
    if (i < 0) {
        i = 0;
    }

    if (seg->n_ramp <= 0 || i >= seg->n_ramp) {
        /* Constant-frequency region (pure tone, or holding past the ramp).
         * Unsigned overflow of the product is exactly phase mod one cycle. */
        const int64_t j = (seg->n_ramp > 0) ? (i - seg->n_ramp) : i;
        const uint64_t base = (seg->n_ramp > 0) ? seg->phase_end_q64 : seg->phase0_q64;
        return base + seg->a_end_q64 * (uint64_t)j;
    }

    uint64_t ph = seg->phase0_q64 + seg->a_q64 * (uint64_t)i;

    if (seg->b_half_q64 != 0) {
        ph += seg->b_half_q64 * (uint64_t)i * (uint64_t)i;
    }

    if (seg->scurve_cyc != 0.0f) {
        /* s-curve = mean-frequency tone minus a bounded sinusoidal deviation:
         *   phi(t) = (f0+f1)/2 * t  -  (df*D/2pi) * sin(pi*t/D)
         * The carrier is folded into a_q64 by the builder, leaving only the
         * bounded term, which float carries comfortably. */
        const double frac = (double)i / (double)seg->n_ramp;
        double dev = -(double)seg->scurve_cyc * sin(AWG_PI * frac);
        dev -= floor(dev); /* reduce to [0,1) before scaling to Q0.64 */
        ph += (uint64_t)(dev * AWG_Q64_SCALE);
    }

    return ph;
}

/* Phase actually emitted: dynamic phase plus this segment's static offset.
 * Use awg_segment_phase_q64 (not this) for the batch-to-batch carry. */
AWG_HD inline uint64_t awg_segment_total_phase_q64(const AwgSegment* seg, int64_t abs_sample) {
    return awg_segment_phase_q64(seg, abs_sample) + seg->static_phase_q64;
}

/* Reduced phase in [0,1) cycles, ready for a sine. */
AWG_HD inline float awg_phase_cycles(uint64_t phase_q64) {
    /* top 32 bits are ample for a float sine argument */
    return (float)(uint32_t)(phase_q64 >> 32) * (float)(1.0 / AWG_Q32_SCALE);
}

/* Build `seg`'s fixed-point coefficients from physical parameters.
 * `duration_s <= 0` builds a pure tone at `f_end_hz`. */
AWG_HD inline void awg_segment_build(AwgSegment* seg, double f_start_hz, double f_end_hz,
                                     double duration_s, int is_scurve, double sample_rate_hz,
                                     int64_t start_sample, uint64_t phase0_q64,
                                     float amplitude) {
    seg->start_sample = start_sample;
    seg->phase0_q64 = phase0_q64;
    seg->amplitude = amplitude;
    seg->b_half_q64 = 0;
    seg->scurve_cyc = 0.0f;

    const int64_t n_ramp =
        (duration_s > 0.0) ? (int64_t)(duration_s * sample_rate_hz + 0.5) : 0;
    seg->n_ramp = n_ramp;
    seg->a_end_q64 = (uint64_t)(f_end_hz / sample_rate_hz * AWG_Q64_SCALE);

    if (n_ramp <= 0) {
        seg->a_q64 = seg->a_end_q64;
        seg->phase_end_q64 = phase0_q64;
        return;
    }

    if (is_scurve) {
        const double f_mid = 0.5 * (f_start_hz + f_end_hz);
        seg->a_q64 = (uint64_t)(f_mid / sample_rate_hz * AWG_Q64_SCALE);
        seg->scurve_cyc = (float)((f_end_hz - f_start_hz) * duration_s / (2.0 * AWG_PI));
    } else {
        seg->a_q64 = (uint64_t)(f_start_hz / sample_rate_hz * AWG_Q64_SCALE);
        const double b = (f_end_hz - f_start_hz) / ((double)n_ramp * sample_rate_hz);
        seg->b_half_q64 = (uint64_t)(int64_t)(0.5 * b * AWG_Q64_SCALE);
    }

    /* Phase at the end of the ramp, so the hold region starts from it. */
    uint64_t pe = phase0_q64 + seg->a_q64 * (uint64_t)n_ramp;
    if (seg->b_half_q64 != 0) {
        pe += seg->b_half_q64 * (uint64_t)n_ramp * (uint64_t)n_ramp;
    }
    seg->phase_end_q64 = pe;
}

#endif /* AWG_ENGINE_PHASE_H */
