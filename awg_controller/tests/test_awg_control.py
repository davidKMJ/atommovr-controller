import unittest

import pytest

from atommovr.algorithms.source._hop import Hop
from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams
from awg_controller.awg_control import AODSettings, RFConverter


class TestAWGControl(unittest.TestCase):
    def setUp(self):
        self.settings = AODSettings(
            f_min_v=80e6,
            f_max_v=89e6,
            f_min_h=80e6,
            f_max_h=89e6,
            grid_rows=10,
            grid_cols=10,
        )
        # AOD_speed 0.1 um/us = 0.1 m/s
        # spacing 5e-6 m = 5 um
        self.params = PhysicalParams(AOD_speed=0.1, spacing=5e-6)
        self.converter = RFConverter(self.settings, self.params)

    def test_grid_to_freq(self):
        self.assertEqual(self.converter._row_to_freq(0), 80e6)
        self.assertEqual(self.converter._col_to_freq(0), 80e6)
        self.assertEqual(self.converter._row_to_freq(1), 81e6)
        self.assertEqual(self.converter._col_to_freq(9), 89e6)

    def test_convert_single_move(self):
        move = Move(0, 0, 1, 1)
        batch = self.converter.convert_moves([move])

        expected_duration = 5e-6 / 0.1
        self.assertAlmostEqual(batch.travel_duration_s, expected_duration)

        expected_ramps = len(self.converter.core_map[0]) + len(
            self.converter.core_map[1]
        )
        self.assertEqual(len(batch.ramps), expected_ramps)

        moved_row = [
            r
            for r in batch.ramps
            if r.channel == 0 and r.f_start == self.converter._row_to_freq(0)
        ][0]
        moved_col = [
            r
            for r in batch.ramps
            if r.channel == 1 and r.f_start == self.converter._col_to_freq(0)
        ][0]
        self.assertEqual(moved_row.f_end, self.converter._row_to_freq(1))
        self.assertEqual(moved_col.f_end, self.converter._col_to_freq(1))

    def test_batch_duration(self):
        moves = [Move(0, 0, 0, 1), Hop(1, 0, 1, 2)]
        with self.assertRaisesRegex(ValueError, "Conflicting column targets"):
            self.converter.convert_moves(moves)

    def test_empty_batch(self):
        batch = self.converter.convert_moves([])
        holding = self.converter.holding_config()
        self.assertEqual(len(batch.ramps), len(holding.ramps))
        self.assertEqual(batch.travel_duration_s, 0.0)


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


if __name__ == "__main__":
    unittest.main()
