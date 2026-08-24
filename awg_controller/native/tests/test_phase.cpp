/* Host-side check of the fixed-point phase against a double reference.
 *
 * Builds with a plain C++ compiler -- no CUDA, no card -- so the numerics can
 * be validated on a laptop:
 *     c++ -O2 -std=c++14 -o test_phase tests/test_phase.cpp && ./test_phase
 *
 * The reference is the same closed form the Python path uses
 * (awg_controller.scapp), evaluated in double and reduced mod 2*pi.
 */

#include "../phase.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static const double kTwoPi = 2.0 * AWG_PI;

/* Double reference: instantaneous phase (radians) of one segment, matching
 * awg_engine.cu's instantaneous_phase()/scapp.py. */
static double ref_phase_rad(double f0, double f1, double D, int is_scurve, double t) {
    if (D <= 0.0) {
        return kTwoPi * f1 * t;
    }
    const double tc = fmin(t, D);
    const double tail = kTwoPi * f1 * fmax(t - D, 0.0);
    if (is_scurve) {
        const double df = f1 - f0;
        return kTwoPi * (f0 * tc + 0.5 * df * (tc - (D / AWG_PI) * sin(AWG_PI * tc / D))) + tail;
    }
    const double slope = (f1 - f0) / D;
    return kTwoPi * (f0 * tc + 0.5 * slope * tc * tc) + tail;
}

static double circ_err(double a, double b) {
    double d = fmod(a - b, kTwoPi);
    if (d < 0) d += kTwoPi;
    if (d > AWG_PI) d = kTwoPi - d;
    return d;
}

static int check(const char* label, double f0, double f1, double D, int is_scurve,
                 double fs, double t_end, int n_probe, double tol_rad) {
    AwgSegment seg;
    awg_segment_build(&seg, f0, f1, D, is_scurve, fs, /*start_sample=*/0,
                      /*phase0_q64=*/0ull, /*amplitude=*/1.0f);

    double worst = 0.0;
    for (int k = 0; k < n_probe; ++k) {
        const double frac = (double)k / (double)(n_probe - 1);
        const int64_t i = (int64_t)(frac * t_end * fs);
        const uint64_t q = awg_segment_phase_q64(&seg, i);
        const double got = (double)q / AWG_Q64_SCALE * kTwoPi;
        const double want = ref_phase_rad(f0, f1, D, is_scurve, (double)i / fs);
        const double e = circ_err(got, want);
        if (e > worst) worst = e;
    }

    const double lsb = worst / kTwoPi * 65536.0; /* phase err as int16 codes */
    const bool ok = worst <= tol_rad;
    printf("  %-34s worst=%9.3e rad  (%8.2f LSB16)  %s\n", label, worst, lsb,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* Same as check(), but evaluated the way the streaming engine actually will:
 * re-basing the segment at every render span so `i` stays bounded. */
static int check_spanned(const char* label, double f0, double f1, double D, int is_scurve,
                         double fs, double t_end, double span_s, double tol_rad) {
    AwgSegment seg;
    awg_segment_build(&seg, f0, f1, D, is_scurve, fs, 0, 0ull, 1.0f);

    const int64_t span = (int64_t)(span_s * fs);
    const int64_t n_end = (int64_t)(t_end * fs);
    double worst = 0.0;

    for (int64_t base = 0; base < n_end; base += span) {
        awg_segment_rebase(&seg, base);
        /* probe a few points inside this span */
        for (int k = 0; k < 8; ++k) {
            const int64_t i = base + (span * k) / 8;
            if (i >= n_end) break;
            const uint64_t q = awg_segment_phase_q64(&seg, i);
            const double got = (double)q / AWG_Q64_SCALE * kTwoPi;
            const double want = ref_phase_rad(f0, f1, D, is_scurve, (double)i / fs);
            const double e = circ_err(got, want);
            if (e > worst) worst = e;
        }
    }

    const double lsb = worst / kTwoPi * 65536.0;
    const bool ok = worst <= tol_rad;
    printf("  %-34s worst=%9.3e rad  (%8.2f LSB16)  %s\n", label, worst, lsb,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* Instantaneous-frequency accuracy: differentiate the fixed-point phase and
 * compare to the intended sweep. This is the quantity that actually positions
 * a trap; a constant phase offset does not move anything. */
static int check_freq(const char* label, double f0, double f1, double D, double fs,
                      double tol_hz) {
    AwgSegment seg;
    awg_segment_build(&seg, f0, f1, D, 0, fs, 0, 0ull, 1.0f);

    double worst = 0.0;
    const int64_t n = (int64_t)(D * fs);
    for (int k = 0; k <= 100; ++k) {
        const int64_t i = (n * k) / 101;
        /* one-sample phase difference -> cycles/sample -> Hz */
        const uint64_t p0 = awg_segment_phase_q64(&seg, i);
        const uint64_t p1 = awg_segment_phase_q64(&seg, i + 1);
        const double dcyc = (double)(uint64_t)(p1 - p0) / AWG_Q64_SCALE;
        const double got_hz = dcyc * fs;
        /* a one-sample difference is the mean frequency over [i, i+1], i.e.
         * the instantaneous frequency at the midpoint -- compare there, or
         * the finite-difference bias b/2 shows up as a fake error. */
        const double want_hz = f0 + (f1 - f0) * (((double)i + 0.5) / (double)n);
        const double e = fabs(got_hz - want_hz);
        if (e > worst) worst = e;
    }
    const bool ok = worst <= tol_hz;
    printf("  %-34s worst=%9.3e Hz  (%.1e relative)  %s\n", label, worst, worst / f1,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(void) {
    const double fs = 1.25e9;
    int bad = 0;

    printf("fixed-point phase vs double reference (fs = %.2f GS/s)\n\n", fs / 1e9);

    printf(" experiment scale (5 us moves):\n");
    bad += check("hold 100 MHz, 5 us", 100e6, 100e6, 0.0, 0, fs, 5e-6, 4096, 1e-4);
    bad += check("linear 60->61 MHz, 5 us", 60e6, 61e6, 5e-6, 0, fs, 5e-6, 4096, 1e-4);
    bad += check("scurve 60->61 MHz, 5 us", 60e6, 61e6, 5e-6, 1, fs, 5e-6, 4096, 1e-4);
    /* Regression: a descending ramp makes the chirp coefficient negative.
     * Stored two's-complement in a uint64, it was previously fed to an
     * *unsigned* 128-bit multiply + >>1, which silently produced garbage --
     * the waveform was off by a full tone amplitude. Ascending ramps and
     * s-curves (b == 0) both looked fine, so only a descending case catches it. */
    bad += check("linear 61->60 MHz, 5 us (descending)", 61e6, 60e6, 5e-6, 0, fs, 5e-6, 4096,
                 1e-4);
    bad += check("linear 100->60 MHz, 3 s (descending)", 100e6, 60e6, 3.0, 0, fs, 3.0, 20000,
                 0.5);

    printf("\n observable scale (the notebook's 3 s demo ramp):\n");
    printf("   absolute phase -- b is quantised to 2^-64, and that error\n");
    printf("   integrates over 6e7 cycles of chirp. Physically a constant\n");
    printf("   phase offset, which does not move a trap, so tolerance is loose:\n");
    bad += check("linear 60->100 MHz, 3 s", 60e6, 100e6, 3.0, 0, fs, 3.0, 20000, 0.5);
    bad += check_spanned("  ... re-based per 1 ms span", 60e6, 100e6, 3.0, 0, fs, 3.0, 1e-3, 0.5);
    printf("   instantaneous frequency -- this is what positions the trap:\n");
    bad += check_freq("linear 60->100 MHz, 3 s", 60e6, 100e6, 3.0, fs, 1.0);
    bad += check_freq("linear 60->61 MHz, 5 us", 60e6, 61e6, 5e-6, fs, 1.0);

    printf("\n indefinite hold (drift is the classic failure):\n");
    bad += check("hold 100 MHz, 10 s", 100e6, 100e6, 0.0, 0, fs, 10.0, 20000, 1e-4);
    bad += check("hold 100 MHz, 60 s", 100e6, 100e6, 0.0, 0, fs, 60.0, 20000, 1e-4);

    printf("\n%s\n", bad ? "FAILURES PRESENT" : "all phase checks passed");
    return bad ? 1 : 0;
}
