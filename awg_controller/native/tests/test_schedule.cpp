/* Host-side check of schedule build: batch-to-batch phase carry and the MEMORY-mode hold tail.
 *
 *     c++ -O2 -std=c++14 -o test_schedule tests/test_schedule.cpp && ./test_schedule
 *
 */

#include "../schedule.h"

#include <cstdio>
#include <vector>

static AWGRoundRamp ramp(int ch, int ti, double f0, double f1, double amp, double deg) {
    AWGRoundRamp rp;
    rp.channel = ch;
    rp.tone_index = ti;
    rp.f_start_hz = f0;
    rp.f_end_hz = f1;
    rp.amplitude_pct = amp;
    rp.phase_deg = deg;
    return rp;
}

/* 2 V tones + 1 H tone; one hold, one move, one zero-duration, one hold. */
static int check_round(int shape) {
    const double fs = 10e6;
    const int nv = 2, nh = 1;
    const double durs[] = {20e-6, 50e-6, 0.0, 30e-6};
    const int32_t counts[] = {3, 3, 3, 3};
    const int nb = 4;

    std::vector<AWGRoundRamp> ramps = {
        ramp(0, 0, 1.0e6, 1.0e6, 30.0, 0.0),
        ramp(0, 1, 1.2e6, 1.2e6, 25.0, 37.0),
        ramp(1, 0, 2.0e6, 2.0e6, 40.0, 0.0),

        ramp(0, 0, 1.0e6, 1.3e6, 30.0, 0.0),
        ramp(0, 1, 1.2e6, 1.1e6, 25.0, 37.0),
        ramp(1, 0, 2.0e6, 2.5e6, 40.0, 12.5),

        ramp(0, 0, 1.3e6, 1.3e6, 30.0, 0.0),
        ramp(0, 1, 1.1e6, 1.1e6, 25.0, 37.0),
        ramp(1, 0, 2.5e6, 2.5e6, 40.0, 12.5),

        ramp(0, 0, 1.3e6, 1.3e6, 30.0, 0.0),
        ramp(0, 1, 1.1e6, 1.1e6, 25.0, 37.0),
        ramp(1, 0, 2.5e6, 2.5e6, 40.0, 12.5),
    };

    AwgSchedule sch;
    char err[256] = {0};
    if (awg_schedule_build(&sch, durs, nb, ramps.data(), (int32_t)ramps.size(), counts, nv,
                           nh, fs, shape, err, sizeof(err)) != 0) {
        fprintf(stderr, "schedule build failed: %s\n", err);
        return 1;
    }

    /* Dynamic phase only: static phase_deg */
    for (int b = 1; b < sch.n_batches; ++b) {
        const int64_t cut = sch.batch_start[b];
        for (int32_t slot = 0; slot < sch.n_tones_total; ++slot) {
            const uint64_t before =
                awg_segment_phase_q64(awg_schedule_segment(&sch, b - 1, slot), cut);
            const uint64_t after =
                awg_segment_phase_q64(awg_schedule_segment(&sch, b, slot), cut);
            if (before != after) {
                fprintf(stderr, "phase carry broken at batch %d slot %d: %llu vs %llu\n", b,
                        slot, (unsigned long long)before, (unsigned long long)after);
                awg_schedule_free(&sch);
                return 1;
            }
        }
    }

    /* MEMORY mode parks by looping this tail; it must meet the round
     * phase-exactly at the seam and contain a whole number of cycles */
    const int64_t tail_samples = 1 << 20;
    AwgSchedule tail;
    if (awg_schedule_hold_tail(&tail, &sch, sch.total_samples, tail_samples, err,
                               sizeof(err)) != 0) {
        fprintf(stderr, "hold tail build failed: %s\n", err);
        awg_schedule_free(&sch);
        return 1;
    }
    for (int32_t slot = 0; slot < sch.n_tones_total; ++slot) {
        const AwgSegment* last = awg_schedule_segment(&sch, sch.n_batches - 1, slot);
        const uint64_t seam_in = awg_segment_total_phase_q64(last, sch.total_samples);
        const uint64_t seam_out = awg_segment_total_phase_q64(&tail.segments[slot], 0);
        const uint64_t wrap = awg_segment_total_phase_q64(&tail.segments[slot], tail_samples);
        if (seam_in != seam_out) {
            fprintf(stderr, "tail seam broken at slot %d: %llu vs %llu\n", slot,
                    (unsigned long long)seam_in, (unsigned long long)seam_out);
            awg_schedule_free(&tail);
            awg_schedule_free(&sch);
            return 1;
        }
        if (wrap != seam_out) {
            fprintf(stderr, "tail does not loop cleanly at slot %d: %llu vs %llu\n", slot,
                    (unsigned long long)wrap, (unsigned long long)seam_out);
            awg_schedule_free(&tail);
            awg_schedule_free(&sch);
            return 1;
        }
    }
    awg_schedule_free(&tail);
    awg_schedule_free(&sch);
    return 0;
}

int main(void) {
    int bad = 0;
    bad += check_round(AWG_ENGINE_SHAPE_LINEAR);
    bad += check_round(AWG_ENGINE_SHAPE_SCURVE);
    printf("%s\n", bad ? "FAILURES PRESENT" : "all schedule checks passed");
    return bad ? 1 : 0;
}
