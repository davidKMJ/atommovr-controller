"""Tests for Recorder as a RoundHook."""

from __future__ import annotations

import json

import numpy as np

from atommovr.algorithms.source._hop import Hop
from atommovr.utils.errormodels import ZeroNoise
from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams

from aod_atommovr.camera import GaussianCameraConfig, OfflineArrayCamera
from aod_atommovr.controller import AodController, HardwareConfig, SoftwareConfig
from aod_atommovr.hooks import RoundContext, SessionContext
from awg_controller.awg_control import AODSettings

from recorder import Recorder, moves_to_records


def _session_ctx(grid_shape=(4, 4), algorithm_name="PCFA", max_rounds=1):
    return SessionContext(
        grid_shape=grid_shape, algorithm_name=algorithm_name, max_rounds=max_rounds
    )


def _round_ctx(
    round_idx=0,
    occupancy=None,
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
        frame=None,
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


class TestRecorder:
    def test_disabled_is_noop(self, tmp_path):
        rec = Recorder(tmp_path, enabled=False)
        rec.on_session_start(_session_ctx())
        assert rec.run_dir is None
        rec.after_round(_round_ctx())
        assert list(tmp_path.iterdir()) == []

    def test_on_session_start_writes_meta(self, tmp_path):
        rec = Recorder(
            tmp_path,
            run_dir=tmp_path / "run_test",
            meta={"grid": [4, 4]},
        )
        rec.on_session_start(_session_ctx())
        assert (rec.run_dir / "meta.json").is_file()
        meta = json.loads((rec.run_dir / "meta.json").read_text())
        assert meta["grid"] == [4, 4]
        assert meta["algorithm_name"] == "PCFA"
        assert (rec.run_dir / "rounds.jsonl").is_file()

    def test_after_round_appends_jsonl(self, tmp_path):
        rec = Recorder(tmp_path, run_dir=tmp_path / "run_jsonl")
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

    def test_after_round_appends_multiple_rounds(self, tmp_path):
        rec = Recorder(tmp_path, run_dir=tmp_path / "run_multi")
        rec.on_session_start(_session_ctx(max_rounds=3))
        for r in range(3):
            rec.after_round(_round_ctx(round_idx=r))

        lines = (rec.run_dir / "rounds.jsonl").read_text().strip().splitlines()
        assert [json.loads(ln)["round"] for ln in lines] == [0, 1, 2]

    def test_after_round_before_session_start_is_noop(self, tmp_path):
        """No run_dir yet -- must not raise, must not write anything."""
        rec = Recorder(tmp_path, run_dir=tmp_path / "run_unstarted")
        rec.after_round(_round_ctx())
        assert not (tmp_path / "run_unstarted").exists()

    def test_moves_to_records(self):
        batches = [[Hop(0, 1, 2, 3), Move(1, 1, 1, 2)]]
        assert moves_to_records(batches) == [
            {"fr": 0, "fc": 1, "tr": 2, "tc": 3},
            {"fr": 1, "fc": 1, "tr": 1, "tc": 2},
        ]
        assert moves_to_records([]) == []


class TestBusCallsRecorderNotController:
    """The controller does not own the recorder; the HookBus calls it."""

    def test_controller_has_no_recorder_attribute(self, tmp_path):
        rows, cols = 6, 5
        rec = Recorder(
            tmp_path,
            run_dir=tmp_path / "ctrl_run",
            meta={"grid": [rows, cols]},
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
        lines = [
            ln
            for ln in (rec.run_dir / "rounds.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(lines) >= 1
        assert ok in (True, False)
