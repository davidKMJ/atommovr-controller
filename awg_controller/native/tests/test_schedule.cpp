/* Renders a round through the fixed-point schedule so it can be diffed
 * against the Python reference (awg_controller.scapp).
 *
 * Reads a round description on argv[1], writes "<ch> <sample> <value>" lines
 * to stdout. Also asserts the batch-to-batch phase carry is continuous.
 *
 *   c++ -O2 -std=c++14 -o test_schedule tests/test_schedule.cpp
 *
 * Driven by tests/compare_with_scapp.py, which authors the round file so both
 * sides are fed byte-identical input.
 */

#include "../schedule.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test_schedule <round-file>\n");
        return 2;
    }
    FILE* f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    double fs;
    int nv, nh, shape, nb;
    if (fscanf(f, "%lf %d %d %d %d", &fs, &nv, &nh, &shape, &nb) != 5) {
        fprintf(stderr, "bad header\n");
        return 2;
    }

    std::vector<double> durs((size_t)nb);
    std::vector<int32_t> counts((size_t)nb);
    std::vector<AWGRoundRamp> ramps;
    for (int b = 0; b < nb; ++b) {
        int nr;
        if (fscanf(f, "%lf %d", &durs[(size_t)b], &nr) != 2) {
            fprintf(stderr, "bad batch %d\n", b);
            return 2;
        }
        counts[(size_t)b] = nr;
        for (int r = 0; r < nr; ++r) {
            AWGRoundRamp rp;
            int ch, ti;
            if (fscanf(f, "%d %d %lf %lf %lf %lf", &ch, &ti, &rp.f_start_hz, &rp.f_end_hz,
                       &rp.amplitude_pct, &rp.phase_deg) != 6) {
                fprintf(stderr, "bad ramp %d/%d\n", b, r);
                return 2;
            }
            rp.channel = ch;
            rp.tone_index = ti;
            ramps.push_back(rp);
        }
    }
    fclose(f);

    AwgSchedule sch;
    char err[256] = {0};
    if (awg_schedule_build(&sch, durs.data(), nb, ramps.data(), (int32_t)ramps.size(),
                           counts.data(), nv, nh, fs, shape, err, sizeof(err)) != 0) {
        fprintf(stderr, "schedule build failed: %s\n", err);
        return 1;
    }

    /* The carry must land the incoming segment exactly where the outgoing one
     * was: a mismatch here is an audible click on the real hardware. */
    for (int b = 1; b < sch.n_batches; ++b) {
        const int64_t cut = sch.batch_start[b];
        for (int32_t slot = 0; slot < sch.n_tones_total; ++slot) {
            const uint64_t before = awg_segment_phase_q64(awg_schedule_segment(&sch, b - 1, slot), cut);
            const uint64_t after = awg_segment_phase_q64(awg_schedule_segment(&sch, b, slot), cut);
            if (before != after) {
                fprintf(stderr, "phase carry broken at batch %d slot %d: %llu vs %llu\n", b,
                        slot, (unsigned long long)before, (unsigned long long)after);
                awg_schedule_free(&sch);
                return 1;
            }
        }
    }

    /* MEMORY mode parks by looping a tail segment forever, so two joins must
     * be exact: round -> tail, and tail -> itself. A break in either is a
     * periodic phase step at the loop rate. */
    {
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
    }

    const double two_pi = 2.0 * AWG_PI;
    for (int64_t s = 0; s < sch.total_samples; ++s) {
        const int32_t b = awg_schedule_batch_at(&sch, s);
        for (int ch = 0; ch < 2; ++ch) {
            double acc = 0.0;
            for (int32_t t = 0; t < sch.n_tones[ch]; ++t) {
                const int32_t slot = awg_tone_slot(&sch, ch, t);
                const AwgSegment* seg = awg_schedule_segment(&sch, b, slot);
                const uint64_t ph = awg_segment_total_phase_q64(seg, s);
                acc += sin((double)ph / AWG_Q64_SCALE * two_pi) * (double)seg->amplitude;
            }
            printf("%d %lld %.17g\n", ch, (long long)s, acc);
        }
    }

    awg_schedule_free(&sch);
    return 0;
}
