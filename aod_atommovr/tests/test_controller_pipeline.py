"""
End-to-end pipeline tests for the atommovr controller.
========================================================

Tests the full pipeline from artificial atom images through rearrangement
algorithm selection, RF conversion, and AWG batch generation — everything
except actual hardware I/O (spcm is not required).

Follows the pattern established in ``test_algorithms.py``:
- Build deterministic atom arrays with known occupancy
- Run algorithm → moves → RF batches
- Verify move correctness, frequency mapping, amplitude budgets, and
  batch structure.
"""

import numpy as np
import pytest

from atommovr.algorithms.single_species import (
    BCv2,
    BalanceAndCompact,
    GeneralizedBalance,
    Hungarian,
    PCFA,
    ParallelHungarian,
    ParallelLBAP,
    Tetris,
)
from atommovr.algorithms.source._hop import Hop
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.Move import Move
from atommovr.utils.core import Configurations, PhysicalParams

from awg_controller.awg_control import (
    AWGBatch,
    MAX_AMPLITUDE_PCT_PER_CHANNEL,
    AODSettings,
    RFConverter,
    RFRamp,
)
from aod_atommovr.camera import OfflineArrayCamera
from aod_atommovr.controller import AodController, HardwareConfig, SoftwareConfig

# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_algorithms.py style)
# ---------------------------------------------------------------------------


def _centered_target_mask(
    array_shape: tuple[int, int],
    target_rows: int,
    target_cols: int | None = None,
) -> np.ndarray:
    """Create a centred rectangular target mask."""
    if target_cols is None:
        target_cols = target_rows
    mask = np.zeros(array_shape, dtype=int)
    r0 = max(0, (array_shape[0] - target_rows) // 2)
    c0 = max(0, (array_shape[1] - target_cols) // 2)
    mask[r0 : r0 + target_rows, c0 : c0 + target_cols] = 1
    return mask


def _default_source_state(
    array_shape: tuple[int, int],
    target_size: int,
) -> np.ndarray:
    """L-shaped border fill with enough atoms to cover the target."""
    rows, cols = array_shape
    t = target_size
    state = np.zeros(array_shape, dtype=int)
    band_rows = min(rows, t + 2)
    band_cols = min(cols, t + 2)
    state[:band_rows, :] = 1
    state[:, :band_cols] = 1
    state[:, -band_cols:] = 1
    return state


def _build_array_with_target(
    grid_rows: int,
    grid_cols: int,
    target_rows: int,
    target_cols: int,
) -> AtomArray:
    """Build an AtomArray with atoms surrounding a centred target region."""
    shape = (grid_rows, grid_cols)
    target = _centered_target_mask(shape, target_rows, target_cols)
    state = _default_source_state(shape, max(target_rows, target_cols))
    # Clear atoms inside the target so the algorithm must move them in
    state[target == 1] = 0

    arr = AtomArray(list(shape), n_species=1)
    arr.matrix[:, :, 0] = state
    arr.target = target.reshape(grid_rows, grid_cols, 1)
    return arr


def _make_simple_settings(
    grid_rows: int = 10,
    grid_cols: int = 5,
) -> AODSettings:
    return AODSettings(
        f_min_v=60e6,
        f_max_v=100e6,
        f_min_h=60e6,
        f_max_h=100e6,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )


def _default_camera(hw: HardwareConfig, sw: SoftwareConfig) -> OfflineArrayCamera:
    """Mirrors the default camera controller.main() builds for the CLI."""
    aod = hw.aod_settings
    return OfflineArrayCamera(
        (aod.grid_rows, aod.grid_cols),
        physical_params=hw.physical_params,
        blob_params=sw.blob_params,
    )


# =====================================================================
# 1. RFConverter unit tests
# =====================================================================


class TestRFConverter:
    """Verify frequency mapping, amplitude budgets, and batch structure."""

    @pytest.fixture
    def converter_10x5(self) -> RFConverter:
        settings = _make_simple_settings(grid_rows=10, grid_cols=5)
        return RFConverter(settings, PhysicalParams())

    @pytest.fixture
    def converter_4x3(self) -> RFConverter:
        settings = _make_simple_settings(grid_rows=4, grid_cols=3)
        return RFConverter(settings, PhysicalParams())

    # -- frequency mapping --

    def test_row_to_freq_boundaries(self, converter_10x5: RFConverter):
        assert converter_10x5._row_to_freq(0) == pytest.approx(60e6)
        assert converter_10x5._row_to_freq(9) == pytest.approx(100e6)

    def test_col_to_freq_boundaries(self, converter_10x5: RFConverter):
        assert converter_10x5._col_to_freq(0) == pytest.approx(60e6)
        assert converter_10x5._col_to_freq(4) == pytest.approx(100e6)

    def test_freq_spacing(self, converter_10x5: RFConverter):
        s = converter_10x5.settings
        expected_v = (100e6 - 60e6) / 9
        expected_h = (100e6 - 60e6) / 4
        assert s.f_spacing_v == pytest.approx(expected_v)
        assert s.f_spacing_h == pytest.approx(expected_h)

    # -- holding config --

    def test_holding_config_ramp_count(self, converter_10x5: RFConverter):
        """Holding config must emit one ramp per grid site."""
        batch = converter_10x5.holding_config()
        assert isinstance(batch, AWGBatch)
        assert len(batch.ramps) == 10 + 5  # grid_rows + grid_cols
        assert batch.travel_duration_s == 0.0

    def test_holding_config_all_static(self, converter_10x5: RFConverter):
        """All ramps in holding config must have f_start == f_end."""
        batch = converter_10x5.holding_config()
        for ramp in batch.ramps:
            assert ramp.f_start == ramp.f_end, (
                f"Core {ramp.core} on ch{ramp.channel}: "
                f"f_start={ramp.f_start} != f_end={ramp.f_end}"
            )

    def test_holding_amplitude_budget(self, converter_10x5: RFConverter):
        """Total amplitude per channel must not exceed 40%."""
        batch = converter_10x5.holding_config()
        ch0_total = sum(r.amplitude_pct for r in batch.ramps if r.channel == 0)
        ch1_total = sum(r.amplitude_pct for r in batch.ramps if r.channel == 1)
        assert ch0_total == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL)
        assert ch1_total == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL)

    def test_holding_uses_correct_cores(self, converter_10x5: RFConverter):
        """Each ramp must use a core from the correct channel."""
        batch = converter_10x5.holding_config()
        core_map = converter_10x5.core_map
        for ramp in batch.ramps:
            assert ramp.core in core_map[ramp.channel], (
                f"Core {ramp.core} not in ch{ramp.channel} map {core_map[ramp.channel]}"
            )

    # -- convert_moves --

    def test_empty_moves_returns_holding(self, converter_10x5: RFConverter):
        batch = converter_10x5.convert_moves([])
        holding = converter_10x5.holding_config()
        assert len(batch.ramps) == len(holding.ramps)
        assert batch.travel_duration_s == 0.0

    def test_single_move_batch_structure(self, converter_4x3: RFConverter):
        """A single move should still include ALL grid tones."""
        move = Move(from_row=0, from_col=0, to_row=1, to_col=1)
        batch = converter_4x3.convert_moves([move])
        # 4 rows + 3 cols = 7 ramps total
        assert len(batch.ramps) == 4 + 3
        assert batch.travel_duration_s > 0

    def test_moving_tone_frequency_changes(self, converter_4x3: RFConverter):
        """Moving tone must have f_end != f_start (unless same row/col)."""
        move = Hop(from_row=0, from_col=0, to_row=2, to_col=1)
        batch = converter_4x3.convert_moves([move])
        # Find the row-0 ramp (ch0, should move to row 2)
        row_ramps = [r for r in batch.ramps if r.channel == 0]
        # The core corresponding to row 0 should have f_end = freq(row 2)
        row0_ramp = row_ramps[0]  # first core = row 0
        assert row0_ramp.f_start == pytest.approx(converter_4x3._row_to_freq(0))
        assert row0_ramp.f_end == pytest.approx(converter_4x3._row_to_freq(2))

    def test_non_moving_tones_stay_static(self, converter_4x3: RFConverter):
        """Non-moving row/col tones must have f_start == f_end."""
        move = Hop(from_row=0, from_col=0, to_row=2, to_col=1)
        batch = converter_4x3.convert_moves([move])
        for ramp in batch.ramps:
            if ramp.channel == 0 and ramp.core != converter_4x3.core_map[0][0]:
                assert ramp.f_start == ramp.f_end, (
                    f"Non-moving ch0 core {ramp.core} changed frequency"
                )

    def test_amplitude_budget_during_moves(self, converter_4x3: RFConverter):
        """Amplitude sum per channel must be exactly 40% during moves."""
        move = Hop(from_row=1, from_col=2, to_row=3, to_col=0)
        batch = converter_4x3.convert_moves([move])
        ch0_total = sum(r.amplitude_pct for r in batch.ramps if r.channel == 0)
        ch1_total = sum(r.amplitude_pct for r in batch.ramps if r.channel == 1)
        assert ch0_total == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL)
        assert ch1_total == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL)

    def test_conflicting_row_targets_raise(self, converter_4x3: RFConverter):
        """Two moves sending the same source row to different target rows."""
        moves = [
            Move(from_row=0, from_col=0, to_row=1, to_col=0),
            Hop(from_row=0, from_col=1, to_row=2, to_col=1),
        ]
        with pytest.raises(ValueError, match="Conflicting row"):
            converter_4x3.convert_moves(moves)

    def test_conflicting_col_targets_raise(self, converter_4x3: RFConverter):
        """Two moves sending the same source col to different target cols."""
        moves = [
            Move(from_row=0, from_col=0, to_row=0, to_col=1),
            Hop(from_row=1, from_col=0, to_row=1, to_col=2),
        ]
        with pytest.raises(ValueError, match="Conflicting column"):
            converter_4x3.convert_moves(moves)

    def test_out_of_bounds_move_raises(self, converter_4x3: RFConverter):
        """Target index beyond grid dimensions should raise."""
        move = Hop(from_row=0, from_col=0, to_row=10, to_col=0)
        with pytest.raises(ValueError, match="out-of-bounds"):
            converter_4x3.convert_moves([move])

    def test_convert_sequence(self, converter_4x3: RFConverter):
        batches = converter_4x3.convert_sequence(
            [
                [Move(0, 0, 1, 1)],
                [Hop(2, 2, 3, 0)],
                [],
            ]
        )
        assert len(batches) == 3
        assert batches[0].travel_duration_s > 0
        assert batches[1].travel_duration_s > 0
        assert batches[2].travel_duration_s == 0  # empty → holding

    # -- core map / tone index --

    def test_core_map_sequential_and_uncapped(self):
        """No fixed tone-count ceiling under SCAPP -- core_map is always
        plain sequential indices, even for a large grid."""
        large = _make_simple_settings(grid_rows=25, grid_cols=8)
        conv = RFConverter(large, PhysicalParams())
        assert conv.core_map[0] == list(range(25))
        assert conv.core_map[1] == list(range(8))

    def test_tone_index_matches_row_col_enumeration(self):
        settings = _make_simple_settings(grid_rows=4, grid_cols=3)
        conv = RFConverter(settings, PhysicalParams())
        holding = conv.holding_config()
        v_ramps = sorted(
            (r for r in holding.ramps if r.channel == 0), key=lambda r: r.tone_index
        )
        h_ramps = sorted(
            (r for r in holding.ramps if r.channel == 1), key=lambda r: r.tone_index
        )
        assert [r.tone_index for r in v_ramps] == list(range(4))
        assert [r.tone_index for r in h_ramps] == list(range(3))


# =====================================================================
# 2. Move duration tests
# =====================================================================


class TestMoveDuration:
    """Verify that move duration is computed from Chebyshev distance."""

    @pytest.fixture
    def converter(self) -> RFConverter:
        settings = _make_simple_settings(grid_rows=10, grid_cols=10)
        params = PhysicalParams()
        return RFConverter(settings, params)

    def test_converter_matches_shared_travel_duration(self, converter: RFConverter):
        """RFConverter travel must equal atommovr.utils.timing.travel_duration_s."""
        from atommovr.utils.timing import travel_duration_s

        move = Hop(0, 0, 2, 3)
        dur = converter._travel_duration_s([move])
        expected = travel_duration_s(
            [move], converter.params.spacing, converter.params.AOD_speed
        )
        assert dur == pytest.approx(expected)

    def test_diagonal_move_chebyshev(self, converter: RFConverter):
        """A (2,3) move should use Chebyshev dist = max(2,3) = 3."""
        from atommovr.utils.timing import travel_duration_s

        move = Hop(0, 0, 2, 3)
        dur = converter._travel_duration_s([move])
        assert dur == pytest.approx(
            travel_duration_s(
                [move], converter.params.spacing, converter.params.AOD_speed
            )
        )

    def test_empty_moves_zero_duration(self, converter: RFConverter):
        assert converter._travel_duration_s([]) == 0.0

    def test_multiple_moves_takes_longest(self, converter: RFConverter):
        """Duration is determined by the move with largest Chebyshev dist."""
        short = Move(0, 0, 1, 0)  # Chebyshev = 1
        long_ = Hop(0, 0, 5, 5)  # Chebyshev = 5
        dur_both = converter._travel_duration_s([short, long_])
        dur_long = converter._travel_duration_s([long_])
        assert dur_both == pytest.approx(dur_long)

    def test_duration_tracks_aod_speed(self):
        """Halving AOD_speed must double travel duration."""
        from atommovr.utils.timing import travel_duration_s

        settings = _make_simple_settings(grid_rows=10, grid_cols=10)
        # spacing is scaled up (vs. the default used elsewhere) so both raw
        # travel times clear MIN_MOVE_DURATION_S with margin -- otherwise
        # the floor would swamp the 2x relationship being tested.
        slow = RFConverter(settings, PhysicalParams(AOD_speed=0.1, spacing=1e-3))
        fast = RFConverter(settings, PhysicalParams(AOD_speed=0.2, spacing=1e-3))
        move = Move(0, 0, 1, 0)
        d_slow = slow.convert_moves([move]).travel_duration_s
        d_fast = fast.convert_moves([move]).travel_duration_s
        assert d_slow == pytest.approx(2 * d_fast)
        assert d_slow == pytest.approx(travel_duration_s([move], 1e-3, 0.1))

    def test_batch_duration_is_travel_only(self):
        """AWG batch duration equals shared travel_duration_s."""
        from atommovr.utils.timing import travel_duration_s

        settings = _make_simple_settings(grid_rows=10, grid_cols=10)
        converter = RFConverter(settings, PhysicalParams(AOD_speed=0.1, spacing=5e-6))
        move = Move(0, 0, 1, 0)
        batch = converter.convert_moves([move])
        assert batch.travel_duration_s == pytest.approx(
            travel_duration_s([move], 5e-6, 0.1)
        )


# =====================================================================
# 3. Full pipeline: algorithm → RF conversion
# =====================================================================

PIPELINE_CASES = [
    {"name": "Hungarian", "cls": Hungarian, "target_size": 4},
    {"name": "PCFA", "cls": PCFA, "target_size": 4},
    {"name": "BCv2", "cls": BCv2, "target_size": 4},
    {"name": "Tetris", "cls": Tetris, "target_size": 4},
    {"name": "BalanceAndCompact", "cls": BalanceAndCompact, "target_size": 4},
]


@pytest.mark.parametrize("case", PIPELINE_CASES, ids=lambda c: c["name"])
class TestAlgorithmToRFPipeline:
    """Run algorithm → convert moves → validate RF batches end to end."""

    def test_rf_batches_respect_amplitude_budget(self, case):
        """Every generated batch must stay within the 40% budget."""
        target_size = case["target_size"]
        grid_rows, grid_cols = target_size + 2, target_size + 2

        settings = _make_simple_settings(grid_rows=grid_rows, grid_cols=grid_cols)
        converter = RFConverter(settings, PhysicalParams())
        arr = _build_array_with_target(grid_rows, grid_cols, target_size, target_size)

        algo = case["cls"]()
        _, move_batches, success = algo.get_moves(arr, do_ejection=False)
        assert success, f"{case['name']} reported failure"

        rf_batches = converter.convert_sequence(move_batches)
        for i, batch in enumerate(rf_batches):
            ch0 = sum(r.amplitude_pct for r in batch.ramps if r.channel == 0)
            ch1 = sum(r.amplitude_pct for r in batch.ramps if r.channel == 1)
            assert ch0 == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL, abs=0.01), (
                f"Batch {i}: ch0 amp {ch0:.2f}% != {MAX_AMPLITUDE_PCT_PER_CHANNEL}%"
            )
            assert ch1 == pytest.approx(MAX_AMPLITUDE_PCT_PER_CHANNEL, abs=0.01), (
                f"Batch {i}: ch1 amp {ch1:.2f}% != {MAX_AMPLITUDE_PCT_PER_CHANNEL}%"
            )

    def test_all_ramps_use_valid_cores(self, case):
        """Every ramp core index must belong to its channel's assignment."""
        target_size = case["target_size"]
        grid_rows, grid_cols = target_size + 2, target_size + 2

        settings = _make_simple_settings(grid_rows=grid_rows, grid_cols=grid_cols)
        converter = RFConverter(settings, PhysicalParams())
        core_map = converter.core_map
        arr = _build_array_with_target(grid_rows, grid_cols, target_size, target_size)

        algo = case["cls"]()
        _, move_batches, success = algo.get_moves(arr, do_ejection=False)
        assert success

        rf_batches = converter.convert_sequence(move_batches)
        for batch in rf_batches:
            for ramp in batch.ramps:
                assert ramp.core in core_map[ramp.channel], (
                    f"Core {ramp.core} not in ch{ramp.channel} map"
                )

    def test_ramp_count_covers_full_grid(self, case):
        """Each batch must emit exactly grid_rows + grid_cols ramps."""
        target_size = case["target_size"]
        grid_rows, grid_cols = target_size + 2, target_size + 2

        settings = _make_simple_settings(grid_rows=grid_rows, grid_cols=grid_cols)
        converter = RFConverter(settings, PhysicalParams())
        arr = _build_array_with_target(grid_rows, grid_cols, target_size, target_size)

        algo = case["cls"]()
        _, move_batches, success = algo.get_moves(arr, do_ejection=False)
        assert success

        rf_batches = converter.convert_sequence(move_batches)
        for batch in rf_batches:
            assert len(batch.ramps) == grid_rows + grid_cols

    def test_frequencies_within_aod_bandwidth(self, case):
        """All generated frequencies must stay within [f_min, f_max]."""
        target_size = case["target_size"]
        grid_rows, grid_cols = target_size + 2, target_size + 2

        settings = _make_simple_settings(grid_rows=grid_rows, grid_cols=grid_cols)
        converter = RFConverter(settings, PhysicalParams())
        arr = _build_array_with_target(grid_rows, grid_cols, target_size, target_size)

        algo = case["cls"]()
        _, move_batches, success = algo.get_moves(arr, do_ejection=False)
        assert success

        rf_batches = converter.convert_sequence(move_batches)
        for batch in rf_batches:
            for ramp in batch.ramps:
                if ramp.channel == 0:
                    assert settings.f_min_v <= ramp.f_end <= settings.f_max_v, (
                        f"V-AOD freq {ramp.f_end / 1e6:.2f} MHz OOB"
                    )
                else:
                    assert settings.f_min_h <= ramp.f_end <= settings.f_max_h, (
                        f"H-AOD freq {ramp.f_end / 1e6:.2f} MHz OOB"
                    )


# =====================================================================
# 4. Controller simulation tests (no spcm hardware)
# =====================================================================


class TestControllerSimulation:
    """Test the AodController in simulation mode (spcm unavailable)."""

    @pytest.fixture
    def controller(self):
        sw = SoftwareConfig(
            algorithm_name="Hungarian",
            max_rounds=3,
        )
        hw = HardwareConfig(
            physical_params=PhysicalParams(middle_size=[4, 3]),
            aod_settings=_make_simple_settings(grid_rows=8, grid_cols=5),
        )
        ctrl = AodController(sw, hw, camera=_default_camera(hw, sw))
        yield ctrl
        ctrl.shutdown()

    def test_controller_creates_rf_converter(self, controller):
        """Controller must initialise an RFConverter with correct core_map."""
        assert controller.rf_converter is not None
        assert 0 in controller.rf_converter.core_map
        assert 1 in controller.rf_converter.core_map

    def test_controller_global_array_target(self, controller):
        """Target is built once at construction onto the global AtomArray."""
        mask = (controller.array.target[:, :, 0] > 0).astype(int)
        assert mask.shape == (8, 5)
        assert mask.sum() == 12  # 4×3 middle fill

    def test_controller_target_zebra_via_target_type(self):
        """Target pattern is resolved once at construction from target_type."""
        sw = SoftwareConfig(
            algorithm_name="Hungarian",
            target_type=Configurations.ZEBRA_HORIZONTAL,
        )
        hw = HardwareConfig(
            physical_params=PhysicalParams(middle_size=[4, 3]),
            aod_settings=_make_simple_settings(grid_rows=8, grid_cols=5),
        )
        ctrl = AodController(sw, hw, camera=_default_camera(hw, sw))
        mask = (ctrl.array.target[:, :, 0] > 0).astype(int)
        assert mask.shape == (8, 5)
        assert np.array_equal(mask[0], np.ones(5, dtype=int))
        assert np.array_equal(mask[1], np.zeros(5, dtype=int))
        assert mask.sum() == 20
        ctrl.shutdown()

    def test_controller_output_batch_simulation(self, controller):
        """_output_batch in sim mode should not raise."""
        holding = controller.rf_converter.holding_config()
        controller._output_batch(holding)  # should log, not crash

    def test_controller_send_holding(self, controller):
        """_send_holding in sim mode should not raise."""
        controller._send_holding()

    def test_controller_rf_converter_holding_matches_grid(self, controller):
        """Holding config ramps should match grid_rows + grid_cols."""
        batch = controller.rf_converter.holding_config()
        s = controller.hw.aod_settings
        assert len(batch.ramps) == s.grid_rows + s.grid_cols

    def test_controller_runs_in_sim_mode(self):
        """Whole-loop smoke test: instantiate, run a short session, shut down."""
        sw = SoftwareConfig(max_rounds=1)
        hw = HardwareConfig(aod_settings=AODSettings(grid_rows=4, grid_cols=3))
        ctrl = AodController(sw, hw, camera=_default_camera(hw, sw))
        try:
            holding = ctrl.rf_converter.holding_config()
            ctrl._output_batch(holding)  # logs + sleeps 0s, must not raise
            ctrl._send_holding()
            ctrl.run()
        finally:
            ctrl.shutdown()


# =====================================================================
# 5. AODSettings validation tests
# =====================================================================


class TestAODSettings:
    def test_frequency_spacing_single_site(self):
        """Grid of 1 shouldn't divide by zero."""
        s = AODSettings(grid_rows=1, grid_cols=1)
        assert s.f_spacing_v == s.f_max_v - s.f_min_v
        assert s.f_spacing_h == s.f_max_h - s.f_min_h

    def test_fov_um_from_um_per_mhz(self):
        s = AODSettings(f_min_v=84.5e6, f_max_v=120.5e6, um_per_mhz=6.526)
        assert s.fov_um_v == pytest.approx(6.526 * 36.0, rel=1e-3)


# =====================================================================
# 6. RFRamp / AWGBatch data class tests
# =====================================================================


class TestDataClasses:
    def test_rf_ramp_defaults(self):
        ramp = RFRamp(channel=0, core=5, f_start=60e6, f_end=70e6, amplitude_pct=4.0)
        assert ramp.phase_deg == 0.0
        assert ramp.duration_s == 0.0
        assert ramp.tone_index == -1

    def test_rf_converter_ramps_carry_real_tone_index(self):
        """RFConverter-produced ramps always carry a real (non-default)
        tone_index matching their row/col enumeration position."""
        settings = _make_simple_settings(grid_rows=4, grid_cols=3)
        conv = RFConverter(settings, PhysicalParams())
        holding = conv.holding_config()
        v_ramps = sorted(
            (r for r in holding.ramps if r.channel == 0), key=lambda r: r.tone_index
        )
        h_ramps = sorted(
            (r for r in holding.ramps if r.channel == 1), key=lambda r: r.tone_index
        )
        assert [r.tone_index for r in v_ramps] == list(range(4))
        assert [r.tone_index for r in h_ramps] == list(range(3))

    def test_awg_batch_construction(self):
        ramps = [
            RFRamp(channel=0, core=i, f_start=60e6, f_end=60e6, amplitude_pct=4.0)
            for i in range(10)
        ]
        batch = AWGBatch(ramps=ramps, travel_duration_s=0.0)
        assert len(batch.ramps) == 10
        assert batch.travel_duration_s == 0.0


# =====================================================================
# 7. Edge-case tests
# =====================================================================


class TestEdgeCases:
    def test_identity_move_produces_static_ramp(self):
        """A move to the same position should produce f_start == f_end."""
        settings = _make_simple_settings(grid_rows=4, grid_cols=3)
        conv = RFConverter(settings, PhysicalParams())
        move = Move(from_row=2, from_col=1, to_row=2, to_col=1)
        batch = conv.convert_moves([move])
        for ramp in batch.ramps:
            assert ramp.f_start == ramp.f_end

    def test_multiple_parallel_moves(self):
        """Multiple non-conflicting parallel moves in one batch."""
        settings = _make_simple_settings(grid_rows=6, grid_cols=5)
        conv = RFConverter(settings, PhysicalParams())
        moves = [
            Move(from_row=0, from_col=0, to_row=1, to_col=1),
            Move(from_row=2, from_col=2, to_row=3, to_col=3),
            Move(from_row=4, from_col=4, to_row=5, to_col=4),
        ]
        batch = conv.convert_moves(moves)
        assert len(batch.ramps) == 6 + 5
        assert batch.travel_duration_s > 0

    def test_same_row_different_col_targets_ok(self):
        """Moves from different rows but sharing a target col is legal
        as long as source cols have consistent targets."""
        settings = _make_simple_settings(grid_rows=4, grid_cols=4)
        conv = RFConverter(settings, PhysicalParams())
        moves = [
            Hop(from_row=0, from_col=0, to_row=2, to_col=1),
            Hop(from_row=1, from_col=0, to_row=3, to_col=1),
        ]
        # Both map col 0 → col 1, which is consistent
        batch = conv.convert_moves(moves)
        assert len(batch.ramps) == 4 + 4

    def test_holding_then_move_then_holding(self):
        """Simulate a round: holding → move batch → holding."""
        settings = _make_simple_settings(grid_rows=4, grid_cols=3)
        conv = RFConverter(settings, PhysicalParams())

        h1 = conv.holding_config()
        move_batch = conv.convert_moves([Move(0, 0, 1, 1)])
        h2 = conv.holding_config()

        # All holding ramps must be static
        for r in h1.ramps:
            assert r.f_start == r.f_end
        for r in h2.ramps:
            assert r.f_start == r.f_end
