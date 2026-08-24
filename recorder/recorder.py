"""Recorder: a simple ``RoundHook`` that writes ``meta.json`` at session
start and appends one JSON line per round to ``rounds.jsonl``. Not part of
the control loop -- attach via ``AtommovrController(..., hooks=[Recorder(...)])``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from atommovr_controller.hooks import RoundContext, SessionContext

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


def moves_to_records(move_batches: Any) -> list[dict[str, int]]:
    """Flatten parallel move batches into compact JSON-serializable dicts."""
    out: list[dict[str, int]] = []
    if not move_batches:
        return out
    for batch in move_batches:
        for m in batch:
            out.append(
                {
                    "fr": int(m.from_row),
                    "fc": int(m.from_col),
                    "tr": int(m.to_row),
                    "tc": int(m.to_col),
                }
            )
    return out


class Recorder:
    """RoundHook: writes ``meta.json`` at session start and appends one
    JSON line per round (atoms/filled/moves/success) to ``rounds.jsonl``.

    Parameters
    ----------
    run_root
        Parent directory; a timestamped ``run_YYYYMMDD_HHMMSS`` folder is
        created underneath at ``on_session_start`` (unless ``run_dir`` is
        passed explicitly).
    enabled
        When ``False``, every method no-ops.
    run_dir
        Optional fixed run directory (tests); skips timestamp folder creation.
    meta
        Optional dict merged into ``meta.json`` at ``on_session_start``.
    """

    def __init__(
        self,
        run_root: PathLike = "runs",
        *,
        enabled: bool = True,
        run_dir: Optional[PathLike] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.run_root = run_root
        self._explicit_run_dir = Path(run_dir) if run_dir is not None else None
        self._meta = dict(meta) if meta else {}
        self.run_dir: Optional[Path] = None
        self._rounds_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # RoundHook
    # ------------------------------------------------------------------

    def on_session_start(self, ctx: SessionContext) -> None:
        if not self.enabled:
            return

        if self._explicit_run_dir is not None:
            self.run_dir = self._explicit_run_dir
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(self.run_root) / f"run_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._rounds_path = self.run_dir / "rounds.jsonl"

        payload: Dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(self.run_dir.resolve()),
            "grid_shape": list(ctx.grid_shape),
            "algorithm_name": ctx.algorithm_name,
            "max_rounds": ctx.max_rounds,
        }
        payload.update(self._meta)
        (self.run_dir / "meta.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        self._rounds_path.touch(exist_ok=True)

    def after_round(self, ctx: RoundContext) -> None:
        """Append one JSON line with this round's stats to ``rounds.jsonl``."""
        if not self.enabled or self._rounds_path is None:
            return

        record: Dict[str, Any] = {
            "round": ctx.round_idx,
            "atoms": ctx.atoms,
            "filled": ctx.filled,
            "need": ctx.need,
            "n_moves": ctx.n_moves,
        }
        if ctx.move_batches is not None:
            record["n_parallel_batches"] = len(ctx.move_batches)
            record["moves"] = moves_to_records(ctx.move_batches)
        if ctx.rf_batches is not None:
            record["n_rf_batches"] = len(ctx.rf_batches)
        if ctx.total_travel_duration_s:
            record["total_travel_duration_s"] = ctx.total_travel_duration_s
        if ctx.success is not None:
            record["success"] = ctx.success
        if ctx.aborted is not None:
            record["aborted"] = ctx.aborted
        with self._rounds_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
