"""
Tests for the SCAPP phase/waveform math (``awg_controller.scapp``) and
the precompute engine's Python control surface
(``awg_controller.awg_engine``).

Pure phase/amplitude math is exercised with plain numpy -- no hardware or
CUDA toolkit required. The C/CUDA precompute engine itself
(``awg_controller/native/awg_engine``) isn't unit-testable without
real hardware; only its Python-side config validation is covered here.
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from awg_controller.awg_control import AODSettings, AWGBatch, RFRamp
from awg_controller.scapp import (
    TWO_PI,
    ToneSegment,
    segment_instantaneous_frequency,
    segment_instantaneous_phase,
    segment_total_phase,
    synthesize_round_frequency_trajectory,
    synthesize_round_waveform,
)
from awg_controller.awg_engine import AWGEngineConfig


def _seg(
    shape,
    f_start,
    f_end,
    duration_s,
    start_sample=0,
    phase_offset_rad=0.0,
    static_phase_rad=0.0,
    amplitude_pct=40.0,
):
    return ToneSegment(
        channel=0,
        tone_index=0,
        shape=shape,
        f_start=f_start,
        f_end=f_end,
        duration_s=duration_s,
        amplitude_pct=amplitude_pct,
        phase_offset_rad=phase_offset_rad,
        static_phase_rad=static_phase_rad,
        start_sample=start_sample,
    )


# =====================================================================
# 1. Pure phase-math tests (hold / linear / scurve)
# =====================================================================


class TestScappSegmentMath:
    def test_hold_phase_matches_constant_frequency(self):
        seg = _seg("hold", f_start=70e6, f_end=70e6, duration_s=0.0)
        t = np.array([0.0, 1e-7, 5e-7, 1e-6])
        phase = segment_instantaneous_phase(seg, t)
        expected = TWO_PI * 70e6 * t
        np.testing.assert_allclose(phase, expected)

    def test_linear_phase_matches_numeric_integration(self):
        f_start, f_end, duration = 70e6, 90e6, 2e-6
        seg = _seg("linear", f_start, f_end, duration)

        def f(tau):
            if tau > duration:
                return f_end
            return f_start + (f_end - f_start) / duration * tau

        for t_val in [
            0.0,
            5e-7,
            1e-6,
            2e-6,
            3e-6,
        ]:  # last two probe the past-duration tail
            expected, _ = quad(f, 0.0, t_val, limit=200)
            expected_phase = TWO_PI * expected
            actual = float(segment_instantaneous_phase(seg, np.array([t_val]))[0])
            assert actual == pytest.approx(expected_phase, rel=1e-6, abs=1e-3)

    def test_scurve_phase_matches_numeric_integration(self):
        f_start, f_end, duration = 70e6, 90e6, 2e-6
        seg = _seg("scurve", f_start, f_end, duration)

        def f(tau):
            if tau > duration:
                return f_end
            return (
                f_start
                + (f_end - f_start) * (1 - math.cos(math.pi * tau / duration)) / 2
            )

        for t_val in [0.0, 5e-7, 1e-6, 2e-6, 3e-6]:
            expected, _ = quad(f, 0.0, t_val, limit=200)
            expected_phase = TWO_PI * expected
            actual = float(segment_instantaneous_phase(seg, np.array([t_val]))[0])
            assert actual == pytest.approx(expected_phase, rel=1e-6, abs=1e-3)

    def test_scurve_matches_linear_when_endpoints_equal(self):
        """Zero-motion segments (f_start == f_end) reduce to a constant
        frequency term regardless of shape (Δf == 0 collapses both ramp
        formulas to the same thing)."""
        seg_lin = _seg("linear", 75e6, 75e6, 1e-6)
        seg_s = _seg("scurve", 75e6, 75e6, 1e-6)
        t = np.array([0.0, 3e-7, 1e-6, 2e-6])
        np.testing.assert_allclose(
            segment_instantaneous_phase(seg_lin, t),
            segment_instantaneous_phase(seg_s, t),
        )

    def test_unknown_shape_raises(self):
        seg = _seg("bogus", 70e6, 70e6, 0.0)
        with pytest.raises(ValueError):
            segment_instantaneous_phase(seg, np.array([0.0]))


class TestSegmentInstantaneousFrequency:
    """segment_instantaneous_frequency is the analytical f(t) counterpart to
    segment_instantaneous_phase — used for the "pure" (FFT-free) frequency
    trajectory panel in SessionRecorder.save_spectrogram.
    """

    def test_hold_is_constant_f_end_regardless_of_t(self):
        seg = _seg("hold", f_start=70e6, f_end=90e6, duration_s=0.0)
        t = np.array([0.0, 1e-6, 5e-6, 1e3])
        np.testing.assert_allclose(segment_instantaneous_frequency(seg, t), 90e6)

    def test_linear_matches_closed_form(self):
        f_start, f_end, duration = 70e6, 90e6, 2e-6
        seg = _seg("linear", f_start, f_end, duration)
        t = np.array([0.0, 5e-7, 1e-6, 2e-6, 3e-6])  # last one past duration
        expected = np.where(
            t <= duration, f_start + (f_end - f_start) * (t / duration), f_end
        )
        np.testing.assert_allclose(
            segment_instantaneous_frequency(seg, t), expected, rtol=1e-9
        )

    def test_scurve_matches_closed_form(self):
        f_start, f_end, duration = 70e6, 90e6, 2e-6
        seg = _seg("scurve", f_start, f_end, duration)
        t = np.array([0.0, 5e-7, 1e-6, 2e-6, 3e-6])
        t_c = np.minimum(t, duration)
        expected = f_start + 0.5 * (f_end - f_start) * (
            1.0 - np.cos(math.pi * t_c / duration)
        )
        np.testing.assert_allclose(
            segment_instantaneous_frequency(seg, t), expected, rtol=1e-9
        )

    @pytest.mark.parametrize("shape", ["linear", "scurve"])
    def test_matches_derivative_of_phase(self, shape):
        """f(t) must be the derivative of segment_instantaneous_phase's
        integral — cross-checked numerically away from the t=0 edge and the
        t=duration kink, where one-sided finite differences are inherently
        imprecise regardless of formula correctness.
        """
        seg = _seg(shape, f_start=70e6, f_end=90e6, duration_s=2e-6)
        t = np.linspace(0.0, 3e-6, 3000)
        dt = t[1] - t[0]
        f_analytic = segment_instantaneous_frequency(seg, t)
        f_numeric = np.gradient(segment_instantaneous_phase(seg, t), dt) / TWO_PI

        kink_idx = int(round(seg.duration_s / dt))
        interior = np.ones(t.shape, dtype=bool)
        interior[:5] = False
        interior[-5:] = False
        interior[max(kink_idx - 5, 0) : kink_idx + 5] = False
        assert np.max(np.abs(f_analytic - f_numeric)[interior]) < 50.0  # Hz

    def test_scurve_matches_linear_when_endpoints_equal(self):
        seg_lin = _seg("linear", 75e6, 75e6, 1e-6)
        seg_s = _seg("scurve", 75e6, 75e6, 1e-6)
        t = np.array([0.0, 3e-7, 1e-6, 2e-6])
        np.testing.assert_allclose(
            segment_instantaneous_frequency(seg_lin, t),
            segment_instantaneous_frequency(seg_s, t),
        )
        np.testing.assert_allclose(segment_instantaneous_frequency(seg_lin, t), 75e6)

    def test_unknown_shape_raises(self):
        seg = _seg("bogus", 70e6, 70e6, 0.0)
        with pytest.raises(ValueError):
            segment_instantaneous_frequency(seg, np.array([0.0]))


class TestSynthesizeRoundFrequencyTrajectory:
    """synthesize_round_frequency_trajectory: the analytical, FFT-free
    per-tone f(t) used for the "pure" frequency trajectory spectrogram panel.
    """

    def test_single_moving_batch(self):
        ramp = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [AWGBatch(ramps=[ramp], travel_duration_s=2e-6)]
        traj = synthesize_round_frequency_trajectory(
            batches, ramp_shape="linear", points_per_batch=5
        )
        assert set(traj) == {(0, 0)}
        times, freqs = traj[(0, 0)]
        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(2e-6)
        assert freqs[0] == pytest.approx(70e6)
        assert freqs[-1] == pytest.approx(90e6)
        assert np.all(np.diff(freqs) >= 0)  # monotonically increasing

    def test_non_moving_ramp_is_flat(self):
        ramp = RFRamp(
            channel=0,
            core=0,
            f_start=80e6,
            f_end=80e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [AWGBatch(ramps=[ramp], travel_duration_s=2e-6)]
        _times, freqs = synthesize_round_frequency_trajectory(batches)[(0, 0)]
        np.testing.assert_allclose(freqs, 80e6)

    def test_cumulative_time_across_batches(self):
        """A hold batch (duration=0) contributes a single point and doesn't
        advance the time offset; subsequent batches' times stack on top of
        prior batches' actual durations.
        """
        ramp1 = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        ramp2 = RFRamp(
            channel=0,
            core=0,
            f_start=90e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        ramp3 = RFRamp(
            channel=0,
            core=0,
            f_start=90e6,
            f_end=70e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [
            AWGBatch(ramps=[ramp1], travel_duration_s=2e-6),
            AWGBatch(ramps=[ramp2], travel_duration_s=0.0),
            AWGBatch(ramps=[ramp3], travel_duration_s=3e-6),
        ]
        times, freqs = synthesize_round_frequency_trajectory(
            batches, points_per_batch=4
        )[(0, 0)]
        assert times[-1] == pytest.approx(5e-6)  # 2e-6 + 0 + 3e-6
        assert freqs.max() == pytest.approx(90e6)
        assert freqs.min() == pytest.approx(70e6)
        assert freqs[-1] == pytest.approx(70e6)  # ends back at 70 MHz

    def test_empty_batches_returns_empty(self):
        assert synthesize_round_frequency_trajectory([]) == {}

    def test_regression_break_inserted_at_genuine_discontinuity(self):
        """RFConverter can rebuild a non-targeted tone's ramp from its
        nominal resting frequency rather than where an earlier batch in the
        same round actually left it — a real commanded frequency jump, not
        a continuous ramp. The trajectory must insert a NaN there so a line
        plot doesn't draw a misleading connecting "drop", instead of
        silently connecting f_end of one batch to a mismatched f_start of
        the next.
        """
        moved_away = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        # Next batch doesn't target tone_index=0: rebuilt as a "hold" at its
        # *nominal* 70 MHz, not the 90 MHz it actually reached above.
        reset_to_nominal = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=70e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [
            AWGBatch(ramps=[moved_away], travel_duration_s=2e-6),
            AWGBatch(ramps=[reset_to_nominal], travel_duration_s=2e-6),
        ]
        times, freqs = synthesize_round_frequency_trajectory(batches)[(0, 0)]
        nan_idx = np.where(np.isnan(freqs))[0]
        assert nan_idx.size == 1
        # The break sits exactly at the batch boundary (t=2e-6), separating
        # the first batch's ramp-up from the second's mismatched restart.
        assert times[nan_idx[0]] == pytest.approx(2e-6)
        assert freqs[nan_idx[0] - 1] == pytest.approx(90e6)  # end of batch 1
        assert freqs[nan_idx[0] + 1] == pytest.approx(70e6)  # start of batch 2

    def test_no_break_when_batches_are_frequency_continuous(self):
        """The normal, well-behaved case (f_start matches the previous
        batch's f_end) must not be broken -- only genuine mismatches are.
        """
        ramp1 = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        ramp2 = RFRamp(
            channel=0,
            core=0,
            f_start=90e6,
            f_end=70e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [
            AWGBatch(ramps=[ramp1], travel_duration_s=2e-6),
            AWGBatch(ramps=[ramp2], travel_duration_s=2e-6),
        ]
        _times, freqs = synthesize_round_frequency_trajectory(batches)[(0, 0)]
        assert not np.any(np.isnan(freqs))

    def test_break_tol_hz_controls_mismatch_threshold(self):
        """A tiny f_start/f_end mismatch within break_tol_hz must not be
        treated as a real discontinuity."""
        ramp1 = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=90e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        ramp2 = RFRamp(
            channel=0,
            core=0,
            f_start=90e6 + 0.1,  # 0.1 Hz off from the true prior f_end
            f_end=70e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [
            AWGBatch(ramps=[ramp1], travel_duration_s=2e-6),
            AWGBatch(ramps=[ramp2], travel_duration_s=2e-6),
        ]
        _times, freqs = synthesize_round_frequency_trajectory(
            batches, break_tol_hz=1.0
        )[(0, 0)]
        assert not np.any(np.isnan(freqs))


def _single_tone_batch(f_start, f_end, duration_s, amplitude_pct=40.0, phase_deg=0.0):
    ramp = RFRamp(
        channel=0,
        core=0,
        f_start=f_start,
        f_end=f_end,
        amplitude_pct=amplitude_pct,
        phase_deg=phase_deg,
        tone_index=0,
    )
    return AWGBatch(ramps=[ramp], travel_duration_s=duration_s)


# =====================================================================
# 2b. Offline waveform synthesis (for spectrogram visualization)
# =====================================================================


class TestSynthesizeRoundWaveform:
    def test_empty_batches_returns_empty_dict(self):
        assert synthesize_round_waveform([], sample_rate_hz=1e6) == {}

    def test_sample_counts_match_durations(self):
        batches = [
            _single_tone_batch(70e6, 70e6, 2e-6),
            _single_tone_batch(70e6, 90e6, 3e-6),
        ]
        waveforms = synthesize_round_waveform(batches, sample_rate_hz=10e6)
        assert set(waveforms) == {0}
        assert waveforms[0].shape == (50,)  # (2+3)us * 10MHz

    def test_hold_tone_is_a_pure_sinusoid_at_f_end(self):
        f = 70e6
        rate = 1e9
        batches = [_single_tone_batch(f, f, 2e-6)]
        waveforms = synthesize_round_waveform(batches, sample_rate_hz=rate)
        samples = waveforms[0]
        t = np.arange(samples.size) / rate
        expected = 0.40 * np.sin(TWO_PI * f * t)
        np.testing.assert_allclose(samples, expected, atol=1e-9)

    def test_multi_channel_batches_produce_one_waveform_per_channel(self):
        ramp0 = RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=80e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        ramp1 = RFRamp(
            channel=1,
            core=0,
            f_start=60e6,
            f_end=65e6,
            amplitude_pct=40.0,
            tone_index=0,
        )
        batches = [AWGBatch(ramps=[ramp0, ramp1], travel_duration_s=1e-6)]
        waveforms = synthesize_round_waveform(batches, sample_rate_hz=10e6)
        assert set(waveforms) == {0, 1}
        assert waveforms[0].shape == waveforms[1].shape == (10,)
        assert not np.allclose(waveforms[0], waveforms[1])

    def test_phase_continuous_across_batch_boundary(self):
        """The waveform's own instantaneous phase must not jump at a batch
        boundary."""
        rate = 1e9
        batches = [
            _single_tone_batch(70e6, 70e6, 1e-6),
            _single_tone_batch(70e6, 90e6, 1e-6),
        ]
        samples = synthesize_round_waveform(batches, sample_rate_hz=rate)[0]
        boundary = int(round(1e-6 * rate))
        # Consecutive-sample deltas should be small everywhere, including
        # right at the boundary — a phase jump would show up as an outlier.
        deltas = np.abs(np.diff(samples))
        assert deltas[boundary - 1] < 5 * np.median(deltas) + 1e-3


# =====================================================================
# 3. Config defaults
# =====================================================================


class TestAWGEngineConfig:
    def test_defaults(self):
        cfg = AWGEngineConfig()
        assert cfg.mode == "stream"
        # notify_samples is the real-time unit of work, not just a DMA
        # tuning knob: one render must fit in notify/sample_rate.
        assert cfg.notify_samples == 262144
        # Ring depth doubles as the render-ahead slack budget, but is capped
        # by the GPU's BAR1 aperture (it is pinned for RDMA). 16M frames is
        # 67 MB, ~13 ms at 1.25 GS/s, and fits a 256 MB BAR1 with room spare.
        assert cfg.dma_buffer_samples == 16 * 1024 * 1024
        assert cfg.dma_buffer_samples % cfg.notify_samples == 0
        assert cfg.fill_start_threshold_promille == 800
        # NOT the card's maximum: an M4i.6631-x8 clocks 1.25 GS/s but its
        # Gen2 x8 link cannot stream the 5.0 GB/s that would need, so the
        # default is a rate the bus can actually sustain.
        assert cfg.sample_rate_hz == 500e6
        assert cfg.ramp_shape == "linear"

    def test_memory_mode_requires_power_of_two_tail(self):
        """The park segment is looped by the card, so it must contain a whole
        number of cycles of every tone -- which the phase arithmetic only
        guarantees for a power-of-two length.
        """
        AWGEngineConfig(mode="memory", hold_tail_samples=1 << 20)
        with pytest.raises(ValueError):
            AWGEngineConfig(mode="memory", hold_tail_samples=1000)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            AWGEngineConfig(mode="precompute")

    def test_dma_buffer_must_be_multiple_of_notify_samples(self):
        """The SCAPP DMA buffer handling only works correctly when
        dma_buffer_samples is an exact multiple of notify_samples --
        enforced here at construction time instead of failing deep inside
        the driver.
        """
        cfg = AWGEngineConfig()  # defaults must satisfy this themselves
        assert cfg.dma_buffer_samples % cfg.notify_samples == 0

        with pytest.raises(ValueError):
            AWGEngineConfig(notify_samples=1000, dma_buffer_samples=32_000_001)

        # A valid, non-default combination must not raise.
        AWGEngineConfig(notify_samples=1024, dma_buffer_samples=1024 * 100)

    def test_notify_samples_must_be_positive(self):
        with pytest.raises(ValueError):
            AWGEngineConfig(notify_samples=0)


# =====================================================================
# 4. Simulation-mode / no-hardware guarantees
# =====================================================================


class TestSimulationGuard:
    def test_scapp_module_imports_without_hardware(self):
        import awg_controller.scapp as sg

        assert sg.ToneSegment is not None
        assert sg.segment_instantaneous_phase is not None

    def test_awg_engine_module_imports_without_native_library(self):
        import awg_controller.awg_engine as sp

        assert sp.AWGEngine is not None
        assert sp.AWGEngineConfig is not None
