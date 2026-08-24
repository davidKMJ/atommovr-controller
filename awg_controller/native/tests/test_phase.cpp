/* Host-side check of the fixed-point phase against a double reference.
 *
 *     c++ -O2 -std=c++14 -o test_phase tests/test_phase.cpp && ./test_phase
 *
 */

#include "../phase.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static const double kTwoPi = 2.0 * AWG_PI;

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

/* Instantaneous-frequency accuracy: differentiate the fixed-point phase and
 * compare to the intended sweep. */
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
    bad += check("linear 61->60 MHz, 5 us", 61e6, 60e6, 5e-6, 0, fs, 5e-6, 4096, 1e-4);
    bad += check_freq("linear 60->61 MHz, 5 us", 60e6, 61e6, 5e-6, fs, 1.0);

    printf("\n observable scale (3 s ramp):\n");
    bad += check("linear 60->100 MHz, 3 s", 60e6, 100e6, 3.0, 0, fs, 3.0, 20000, 0.5);
    bad += check_freq("linear 60->100 MHz, 3 s", 60e6, 100e6, 3.0, fs, 1.0);

    printf("\n indefinite hold:\n");
    bad += check("hold 100 MHz, 10 s", 100e6, 100e6, 0.0, 0, fs, 10.0, 20000, 1e-4);
    bad += check("hold 100 MHz, 60 s", 100e6, 100e6, 0.0, 0, fs, 60.0, 20000, 1e-4);

    printf("\n%s\n", bad ? "FAILURES PRESENT" : "all phase checks passed");
    return bad ? 1 : 0;
}
