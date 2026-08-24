import pytest

from atommovr.algorithms.source._hop import Hop
from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams
from awg_controller.awg_control import AODSettings, RFConverter


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
