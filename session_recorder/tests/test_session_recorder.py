"""Tests for SessionRecorder as a RoundHook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from atommovr.algorithms.source._hop import Hop
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.errormodels import ZeroNoise
from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams

from aod_atommovr.camera import GaussianCameraConfig, OfflineArrayCamera
from aod_atommovr.controller import AodController, HardwareConfig, SoftwareConfig
from aod_atommovr.hooks import RoundContext, SessionContext
from awg_controller.awg_control import AODSettings

from session_recorder import GifOptions, SessionRecorder, VisualizationOptions, moves_to_records


def _session_ctx(grid_shape=(4, 4), algorithm_name="PCFA", max_rounds=1):
    return SessionContext(
        grid_shape=grid_shape, algorithm_name=algorithm_name, max_rounds=max_rounds
    )


def _round_ctx(
    round_idx=0,
    occupancy=None,
    frame=None,
    move_batches=None,
    rf_batches=None,
    success=None,
    aborted=None,
):
    occ = np.eye(4, dtype=int) if occupancy is None else occupancy
    ctx = RoundContext(
        round_idx=round_idx,
        occupancy=occ,
        target=occ,
        frame=frame,
        atoms=int(occ.sum()),
        filled=0,
        need=int(occ.sum()),
    )
    if move_batches is not None:
        ctx.move_batches = move_batches
        ctx.n_moves = sum(len(b) for b in move_batches)
    ctx.rf_batches = rf_batches
    ctx.success = success
    ctx.aborted = aborted
    return ctx


class TestSessionRecorder:
    def test_disabled_is_noop(self, tmp_path):
        rec = SessionRecorder(tmp_path, enabled=False)
        rec.on_session_start(_session_ctx())
        assert rec.run_dir is None
        rec.before_round(_round_ctx())
        rec.after_round(_round_ctx())
        assert list(tmp_path.iterdir()) == []

    def test_before_round_dumps_frame_and_occupancy(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "run_test",
            meta={"grid": [4, 4]},
            gif=GifOptions(enabled=False),
        )
        rec.on_session_start(_session_ctx())
        assert (rec.run_dir / "meta.json").is_file()
        meta = json.loads((rec.run_dir / "meta.json").read_text())
        assert meta["grid"] == [4, 4]

        frame = np.arange(16, dtype=np.uint8).reshape(4, 4)
        occ = np.eye(4, dtype=int)
        rec.before_round(_round_ctx(round_idx=0, occupancy=occ, frame=frame))

        round_dir = rec.run_dir / "round_00"
        assert round_dir.is_dir()
        assert (round_dir / "frame.npy").is_file()
        assert (round_dir / "occupancy.npy").is_file()
        assert np.array_equal(np.load(round_dir / "occupancy.npy"), occ)

    def test_after_round_appends_jsonl(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "run_jsonl",
            gif=GifOptions(enabled=False),
        )
        rec.on_session_start(_session_ctx())
        ctx = _round_ctx(
            round_idx=0,
            move_batches=[[Move(0, 0, 1, 1)]],
            success=True,
        )
        ctx.filled = ctx.need
        rec.after_round(ctx)

        lines = (rec.run_dir / "rounds.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["round"] == 0
        assert row["n_moves"] == 1
        assert row["moves"][0]["tr"] == 1
        assert row["success"] is True

    def test_moves_to_records(self):
        batches = [[Hop(0, 1, 2, 3), Move(1, 1, 1, 2)]]
        assert moves_to_records(batches) == [
            {"fr": 0, "fc": 1, "tr": 2, "tc": 3},
            {"fr": 1, "fc": 1, "tr": 1, "tc": 2},
        ]
        assert moves_to_records([]) == []

    def test_gif_options_written_to_meta(self, tmp_path):
        gif = GifOptions(
            enabled=True,
            sources=("occupancy",),
            duration_s=0.25,
            max_side=64,
            auto_write=False,
        )
        rec = SessionRecorder(tmp_path, run_dir=tmp_path / "gif_meta", gif=gif)
        rec.on_session_start(_session_ctx())
        meta = json.loads((rec.run_dir / "meta.json").read_text())
        assert meta["gif"]["sources"] == ["occupancy"]
        assert meta["gif"]["duration_s"] == 0.25
        assert meta["gif"]["auto_write"] is False

    def test_write_gifs_from_round_dumps(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "gif_run",
            gif=GifOptions(
                enabled=True,
                sources=("frame", "occupancy"),
                duration_s=0.2,
                max_side=128,
                occupancy_cell_px=8,
                auto_write=False,
            ),
        )
        rec.on_session_start(_session_ctx())
        for r in range(3):
            rec.before_round(
                _round_ctx(
                    round_idx=r,
                    frame=np.full((32, 32), 50 + r * 40, dtype=np.uint8),
                    occupancy=(np.arange(16).reshape(4, 4) > r).astype(int),
                )
            )

        written = rec.finalize()
        assert "frame" in written
        assert "occupancy" in written
        assert (rec.run_dir / "frames.gif").is_file()
        assert (rec.run_dir / "occupancy.gif").is_file()
        assert len(rec._gif_frames["frame"]) == 3
        assert len(rec._gif_frames["occupancy"]) == 3

    def test_gif_disabled_skips_files(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "no_gif",
            gif=GifOptions(enabled=False),
        )
        rec.on_session_start(_session_ctx())
        rec.before_round(
            _round_ctx(
                round_idx=0,
                frame=np.zeros((8, 8), dtype=np.uint8),
                occupancy=np.ones((2, 2), dtype=int),
            )
        )
        assert rec.finalize() == {}
        assert not (rec.run_dir / "frames.gif").exists()
        assert not (rec.run_dir / "occupancy.gif").exists()


def _one_move_batches(n=1):
    return [[Move(0, 0, 1, 1)] for _ in range(n)]


class TestMoveVisualization:
    def test_disabled_by_default_is_noop(self, tmp_path):
        rec = SessionRecorder(tmp_path, run_dir=tmp_path / "viz_off")
        rec.on_session_start(_session_ctx())
        rec.after_round(_round_ctx(move_batches=_one_move_batches()))
        assert not (rec.run_dir / "round_00_visualization").exists()

    def test_empty_move_batches_is_noop(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_empty",
            visualization=VisualizationOptions(enabled=True),
        )
        rec.on_session_start(_session_ctx())
        rec.after_round(_round_ctx(move_batches=[]))
        assert not (rec.run_dir / "round_00_visualization").exists()

    def test_writes_grid_and_gif_by_default(self, tmp_path, monkeypatch):
        """gif defaults to True -- both the static grid.svg and the
        animated grid.gif must be written."""
        import aod_atommovr.imaging.visualization as viz_mod

        calls = []

        def _fake_grid_tracking(
            occupancy, batches, save_path=None, title_suffix="", max_cols=3
        ):
            calls.append(len(batches))
            Path(save_path).write_text("x")

        def _fake_frames(occupancy, batches, **kwargs):
            calls.append(len(batches))
            # 3x3x3 uint8 "frames" -- enough for _write_gif to accept.
            return [
                np.zeros((3, 3, 3), dtype=np.uint8) for _ in range(len(batches) + 1)
            ]

        monkeypatch.setattr(viz_mod, "visualize_move_batches", _fake_grid_tracking)
        monkeypatch.setattr(viz_mod, "render_move_batch_frames", _fake_frames)

        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_both",
            visualization=VisualizationOptions(enabled=True),
        )
        rec.on_session_start(_session_ctx())
        rec.after_round(_round_ctx(round_idx=3, move_batches=_one_move_batches(2)))
        stage = rec.run_dir / "round_03_visualization"
        assert stage.is_dir()
        assert (stage / "grid.svg").is_file()
        assert (stage / "grid.gif").is_file()
        assert calls == [2, 2]

    def test_gif_false_skips_gif_and_frame_rendering(self, tmp_path, monkeypatch):
        import aod_atommovr.imaging.visualization as viz_mod

        def _fake_grid(occupancy, batches, save_path=None, title_suffix="", max_cols=3):
            Path(save_path).write_text("x")

        def _boom(*args, **kwargs):
            raise AssertionError("render_move_batch_frames must not be called")

        monkeypatch.setattr(viz_mod, "visualize_move_batches", _fake_grid)
        monkeypatch.setattr(viz_mod, "render_move_batch_frames", _boom)

        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_no_gif",
            visualization=VisualizationOptions(enabled=True, gif=False),
        )
        rec.on_session_start(_session_ctx())
        rec.after_round(_round_ctx(round_idx=0, move_batches=_one_move_batches(1)))
        stage = rec.run_dir / "round_00_visualization"
        assert (stage / "grid.svg").is_file()
        assert not (stage / "grid.gif").exists()

    def test_regression_max_batches_truncates(self, tmp_path, monkeypatch):
        """A round with many small parallel-move batches would otherwise
        render one enormous, unreadable figure -- max_batches must cap how
        many batches actually get visualized (both grid and gif)."""
        import aod_atommovr.imaging.visualization as viz_mod

        captured = {}

        def _fake_grid_tracking(
            occupancy, batches, save_path=None, title_suffix="", max_cols=3
        ):
            captured["grid_n_batches"] = len(batches)
            Path(save_path).write_text("x")

        def _fake_frames(occupancy, batches, **kwargs):
            captured["gif_n_batches"] = len(batches)
            return [
                np.zeros((3, 3, 3), dtype=np.uint8) for _ in range(len(batches) + 1)
            ]

        monkeypatch.setattr(viz_mod, "visualize_move_batches", _fake_grid_tracking)
        monkeypatch.setattr(viz_mod, "render_move_batch_frames", _fake_frames)

        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_trunc",
            visualization=VisualizationOptions(enabled=True, max_batches=3),
        )
        rec.on_session_start(_session_ctx())
        rec.after_round(_round_ctx(round_idx=0, move_batches=_one_move_batches(10)))
        assert captured["grid_n_batches"] == 3
        assert captured["gif_n_batches"] == 3

    def test_max_batches_none_disables_cap(self, tmp_path, monkeypatch):
        import aod_atommovr.imaging.visualization as viz_mod

        captured = {}

        def _fake_grid_tracking(
            occupancy, batches, save_path=None, title_suffix="", max_cols=3
        ):
            captured["n_batches"] = len(batches)
            Path(save_path).write_text("x")

        monkeypatch.setattr(viz_mod, "visualize_move_batches", _fake_grid_tracking)

        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_no_cap",
            visualization=VisualizationOptions(
                enabled=True, gif=False, max_batches=None
            ),
        )
        rec.on_session_start(_session_ctx())
        rec.after_round(_round_ctx(round_idx=0, move_batches=_one_move_batches(10)))
        assert captured["n_batches"] == 10

    def test_regression_real_rendering_small_array(self, tmp_path):
        """End-to-end with the real aod_atommovr.imaging.visualization
        functions (no mocking) on a tiny occupancy/round, to catch
        integration issues (signature mismatches, etc.) a mocked test
        can't -- checks both the static figure and the gif.
        """
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_real",
            visualization=VisualizationOptions(enabled=True),
        )
        rec.on_session_start(_session_ctx())
        occ = np.ones((3, 3), dtype=int)
        rec.after_round(
            _round_ctx(round_idx=0, occupancy=occ, move_batches=_one_move_batches(2))
        )
        stage = rec.run_dir / "round_00_visualization"
        grid_path = stage / "grid.svg"
        gif_path = stage / "grid.gif"
        assert grid_path.is_file()
        assert grid_path.stat().st_size > 0
        assert gif_path.is_file()
        assert gif_path.stat().st_size > 0


class TestBeforeRoundDumpHasFrameAndOccupancy:
    """Replaces the old acquire/detect stage-folder test: a real camera
    sync feeds the controller's before_round context, which the recorder
    dumps as one round folder with both frame + occupancy."""

    def test_sync_then_before_round_dumps_frame_and_occupancy(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "cam_run",
            gif=GifOptions(enabled=False),
        )
        occ0 = np.ones((4, 4), dtype=int)
        cam = OfflineArrayCamera(
            (4, 4),
            image_generator=GaussianCameraConfig(
                image_shape=(128, 128), min_spacing_px=16.0, noise_level=0.0
            ),
            initial_occupancy=occ0,
            seed=0,
        )
        array = AtomArray((4, 4), n_species=1, error_model=ZeroNoise(seed=0))
        cam.sync(array)

        rec.on_session_start(_session_ctx(grid_shape=(4, 4)))
        state = (array.matrix[:, :, 0] > 0).astype(int)
        ctx = _round_ctx(round_idx=0, occupancy=state, frame=cam.last_frame)
        rec.before_round(ctx)

        round_dir = rec.run_dir / "round_00"
        assert round_dir.is_dir()
        assert (round_dir / "frame.npy").is_file()
        assert (round_dir / "occupancy.npy").is_file()
        det = np.load(round_dir / "occupancy.npy")
        assert det.shape == (4, 4)
        assert int(det.sum()) == 16


class TestBusCallsRecorderNotController:
    """The controller does not own the recorder; the HookBus calls it."""

    def test_controller_has_no_recorder_attribute(self, tmp_path):
        rows, cols = 6, 5
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "ctrl_run",
            meta={"grid": [rows, cols]},
            gif=GifOptions(
                enabled=True,
                sources=("frame", "occupancy"),
                duration_s=0.15,
                max_side=128,
                auto_write=False,
            ),
        )
        cam = OfflineArrayCamera(
            (rows, cols),
            image_generator=GaussianCameraConfig(
                image_shape=(200, 200), min_spacing_px=20.0, noise_level=0.0
            ),
            physical_params=PhysicalParams(loading_prob=0.9, spacing=5e-6),
            seed=3,
        )
        sw = SoftwareConfig(
            max_rounds=2,
            algorithm_name="Hungarian",
            error_model=ZeroNoise(seed=3),
        )
        hw = HardwareConfig(
            physical_params=PhysicalParams(
                loading_prob=0.9, spacing=5e-6, middle_size=[2, 2]
            ),
            aod_settings=AODSettings(grid_rows=rows, grid_cols=cols),
        )
        with AodController(sw, hw, camera=cam, hooks=[rec]) as ctrl:
            assert not hasattr(ctrl, "recorder")
            ok = ctrl.run()

        assert (rec.run_dir / "meta.json").is_file()
        assert (rec.run_dir / "rounds.jsonl").is_file()
        assert (rec.run_dir / "round_00" / "frame.npy").is_file()
        assert (rec.run_dir / "round_00" / "occupancy.npy").is_file()
        # on_session_end (via the bus) calls recorder.finalize() in the
        # controller's run() finally block.
        assert (rec.run_dir / "frames.gif").is_file()
        assert (rec.run_dir / "occupancy.gif").is_file()
        lines = [
            ln
            for ln in (rec.run_dir / "rounds.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(lines) >= 1
        assert ok in (True, False)
