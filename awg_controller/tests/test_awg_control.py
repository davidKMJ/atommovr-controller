import numpy as np
import pytest

from atommovr.algorithms.source._hop import Hop
from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams
from awg_controller.awg_control import (
    MAX_AMPLITUDE_PCT_PER_CHANNEL,
    REFERENCE_FREQUENCY_HZ,
    AmplitudeCompensation,
    AODSettings,
    RFConverter,
)


@pytest.fixture
def converter():
    settings = AODSettings(
        f_min_v=80e6,
        f_max_v=89e6,
        f_min_h=80e6,
        f_max_h=89e6,
        grid_rows=10,
        grid_cols=10,
    )
    # AOD_speed 0.1 um/us = 0.1 m/s
    # spacing 5e-6 m = 5 um
    params = PhysicalParams(AOD_speed=0.1, spacing=5e-6)
    return RFConverter(settings, params)


class TestAWGControl:
    def test_grid_to_freq(self, converter):
        assert converter._row_to_freq(0) == 80e6
        assert converter._col_to_freq(0) == 80e6
        assert converter._row_to_freq(1) == 81e6
        assert converter._col_to_freq(9) == 89e6

    def test_convert_single_move(self, converter):
        move = Move(0, 0, 1, 1)
        batch = converter.convert_moves([move])

        expected_duration = 5e-6 / 0.1
        assert batch.travel_duration_s == pytest.approx(expected_duration)

        expected_ramps = converter.settings.grid_rows + converter.settings.grid_cols
        assert len(batch.ramps) == expected_ramps

        moved_row = [
            r
            for r in batch.ramps
            if r.channel == 0 and r.f_start == converter._row_to_freq(0)
        ][0]
        moved_col = [
            r
            for r in batch.ramps
            if r.channel == 1 and r.f_start == converter._col_to_freq(0)
        ][0]
        assert moved_row.f_end == converter._row_to_freq(1)
        assert moved_col.f_end == converter._col_to_freq(1)

    def test_batch_duration(self, converter):
        moves = [Move(0, 0, 0, 1), Hop(1, 0, 1, 2)]
        with pytest.raises(ValueError, match="Conflicting column targets"):
            converter.convert_moves(moves)

    def test_empty_batch(self, converter):
        batch = converter.convert_moves([])
        holding = converter.holding_config()
        assert len(batch.ramps) == len(holding.ramps)
        assert batch.travel_duration_s == 0.0


class TestAmplitudeCompensation:
    """Non-default: amplitude as a function of frequency."""

    def test_default_is_equal_power(self, converter):
        holding = converter.holding_config()
        for r in holding.ramps:
            assert r.amplitude_pct == pytest.approx(
                MAX_AMPLITUDE_PCT_PER_CHANNEL / converter.settings.grid_rows
            )
            assert r.amplitude_comp_mode == 0

    def test_reference_amplitude_required(self):
        settings = AODSettings(grid_rows=3, grid_cols=3)
        comp = AmplitudeCompensation("linear", a=1.0, b=0.0)
        with pytest.raises(ValueError, match="reference_amplitude_pct"):
            RFConverter(settings, PhysicalParams(), amplitude_compensation=comp)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            AmplitudeCompensation("bogus", a=1.0, b=0.0)

    def test_no_renormalization(self):
        settings = AODSettings(
            f_min_v=80e6, f_max_v=89e6, f_min_h=80e6, f_max_h=89e6,
            grid_rows=10, grid_cols=10,
        )
        comp = AmplitudeCompensation("linear", a=1.0, b=0.0)  # ratio(f) == 1 everywhere
        conv = RFConverter(
            settings, PhysicalParams(), amplitude_compensation=comp, reference_amplitude_pct=3.0
        )
        holding = conv.holding_config()
        ch0 = [r for r in holding.ramps if r.channel == 0]
        for r in ch0:
            assert r.amplitude_pct == pytest.approx(3.0)
        # Not renormalized up to the 40% budget.
        assert sum(r.amplitude_pct for r in ch0) == pytest.approx(30.0)

    def test_exceeding_budget_raises(self):
        settings = AODSettings(grid_rows=10, grid_cols=10)
        comp = AmplitudeCompensation("linear", a=1.0, b=0.0)
        conv = RFConverter(
            settings, PhysicalParams(), amplitude_compensation=comp, reference_amplitude_pct=5.0
        )
        with pytest.raises(ValueError, match="per-channel budget"):
            conv.holding_config()

    def test_non_positive_amplitude_raises(self):
        settings = AODSettings(grid_rows=3, grid_cols=3, f_min_v=-1e6, f_max_v=1e6)
        comp = AmplitudeCompensation("linear", a=0.0, b=1.0)  # ratio(0 Hz) == 0
        conv = RFConverter(
            settings, PhysicalParams(), amplitude_compensation=comp, reference_amplitude_pct=1.0
        )
        with pytest.raises(ValueError, match="positive amplitudes"):
            conv.holding_config()

    def test_ramps_carry_compensation_params_for_native(self):
        settings = AODSettings(
            f_min_v=80e6, f_max_v=89e6, f_min_h=80e6, f_max_h=89e6,
            grid_rows=10, grid_cols=10,
        )
        comp = AmplitudeCompensation("gaussian", a=1.2, b=0.6, f0_hz=100e6, sigma_hz=15e6)
        conv = RFConverter(
            settings,
            PhysicalParams(AOD_speed=0.1, spacing=5e-6),
            amplitude_compensation=comp,
            reference_amplitude_pct=3.0,
        )
        batch = conv.convert_moves([Move(0, 0, 1, 0)])
        for r in batch.ramps:
            assert r.amplitude_comp_mode == 2  # gaussian
            assert r.amplitude_comp_a == pytest.approx(1.2)
            assert r.amplitude_comp_b == pytest.approx(0.6)
            assert r.amplitude_comp_f0_hz == pytest.approx(100e6)
            assert r.amplitude_comp_sigma_hz == pytest.approx(15e6)
            assert r.amplitude_reference_pct == pytest.approx(3.0)


class TestAmplitudeCompensationFit:
    def test_fit_linear_reference_is_one(self):
        freqs = np.linspace(80e6, 121e6, 20)
        powers = 2.0 + 3e-9 * freqs
        comp = AmplitudeCompensation.fit_linear(freqs, powers)
        assert comp.mode == "linear"
        assert comp(REFERENCE_FREQUENCY_HZ) == pytest.approx(1.0)
        assert comp(freqs[-1]) > comp(freqs[0])

    def test_fit_gaussian_reference_is_one_and_recovers_dip(self):
        freqs = np.linspace(80e6, 121e6, 40)
        true_f0, true_sigma = 100e6, 10e6
        powers = 1.0 - 0.3 * np.exp(-((freqs - true_f0) ** 2) / (2 * true_sigma**2))
        comp = AmplitudeCompensation.fit_gaussian(freqs, powers)
        assert comp.mode == "gaussian"
        assert comp(REFERENCE_FREQUENCY_HZ) == pytest.approx(1.0, abs=1e-6)
        assert comp.f0_hz == pytest.approx(true_f0, rel=0.05)


class TestAODSettingsMapping:
    """Imaging assigns sites; RF maps those indices with f_min/f_max only."""

    def test_index_linear_matches_imaging_contract(self):
        settings = AODSettings(
            grid_rows=5,
            grid_cols=5,
            f_min_v=60e6,
            f_max_v=100e6,
            f_min_h=70e6,
            f_max_h=110e6,
        )
        rf = RFConverter(settings, PhysicalParams())
        assert rf._row_to_freq(0) == pytest.approx(60e6)
        assert rf._row_to_freq(4) == pytest.approx(100e6)
        assert rf._col_to_freq(0) == pytest.approx(70e6)
        assert rf._col_to_freq(4) == pytest.approx(110e6)

    def test_um_per_mhz_fov_ballpark(self):
        settings = AODSettings(
            f_min_v=84.5e6,
            f_max_v=120.5e6,
            um_per_mhz=6.526,
        )
        assert settings.fov_um_v == pytest.approx(6.526 * 36.0, rel=1e-3)
