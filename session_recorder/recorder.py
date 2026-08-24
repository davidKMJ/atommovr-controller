"""SessionRecorder: a bundled ``RoundHook``, not part of the control loop.

Off by default in the sense that a controller only calls it if it's passed
into ``hooks=``; when ``enabled=False`` it no-ops throughout. Writes:

* one frame/occupancy dump per round: ``round_{rr:02d}/frame.png``,
  ``occupancy.npy``, … (from ``before_round`` -- there is no separate
  acquire/detect stage split any more, since a ``RoundContext`` only ever
  carries one already-measured frame + occupancy per round)
* append-only ``rounds.jsonl`` for move / RF statistics (from ``after_round``)
* optional lattice visualization (``grid.<fmt>`` / ``grid.gif``) built from
  ``RoundContext.occupancy`` + ``RoundContext.move_batches`` -- a purely
  geometric replay, since a hook never sees the live, physics-aware
  ``AtomArray`` (see ``aod_atommovr.hooks`` for why)
* optional GIFs (``frames.gif`` / ``occupancy.gif``) finalized at
  ``on_session_end``
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from aod_atommovr.hooks import RoundContext, SessionContext

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an array to uint8 grayscale (or pass through HxWx3 uint8)."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr
    amin = float(np.min(arr)) if arr.size else 0.0
    amax = float(np.max(arr)) if arr.size else 1.0
    if amax <= amin:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    scaled = (arr.astype(np.float64) - amin) / (amax - amin)
    return (scaled * 255.0).astype(np.uint8)


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write a 2-D grayscale (or HxWx{1,3}) array as PNG via OpenCV if present."""
    arr = _to_uint8(image)
    try:
        import cv2  # soft dependency

        ok = cv2.imwrite(str(path), arr)
        if not ok:
            raise OSError(f"cv2.imwrite failed for {path}")
    except ImportError:
        # Fallback: raw dump already kept as .npy; skip PNG.
        pass


def _occupancy_heatmap(occ: np.ndarray, cell_px: int = 16) -> np.ndarray:
    """Upsample a binary occupancy grid to a small uint8 image."""
    binary = (np.asarray(occ) > 0).astype(np.uint8) * 255
    if cell_px <= 1:
        return binary
    return np.kron(binary, np.ones((cell_px, cell_px), dtype=np.uint8))


def _resize_max_side(image: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    """Downscale so max(H, W) <= max_side (nearest-neighbour)."""
    if max_side is None or max_side <= 0:
        return image
    h, w = image.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return image
    scale = max_side / float(m)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    try:
        import cv2

        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_NEAREST)
    except ImportError:
        ys = (np.linspace(0, h - 1, nh)).astype(int)
        xs = (np.linspace(0, w - 1, nw)).astype(int)
        return image[ys][:, xs]


def _write_gif(
    path: Path,
    frames: Sequence[np.ndarray],
    *,
    duration_s: float,
    loop: int,
) -> bool:
    """Write a GIF from uint8 frames. Returns True on success."""
    if not frames:
        return False
    duration_s = max(float(duration_s), 1e-3)
    # Ensure shared shape (crop/pad to first frame).
    h0, w0 = frames[0].shape[:2]
    normed = []
    for fr in frames:
        fr = _to_uint8(fr)
        if fr.ndim == 2:
            canvas = np.zeros((h0, w0), dtype=np.uint8)
        else:
            canvas = np.zeros((h0, w0, fr.shape[2]), dtype=np.uint8)
        hh, ww = min(h0, fr.shape[0]), min(w0, fr.shape[1])
        canvas[:hh, :ww] = fr[:hh, :ww]
        normed.append(canvas)

    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            str(path),
            mode="I",
            duration=duration_s,
            loop=int(loop),
        ) as writer:
            for fr in normed:
                writer.append_data(fr)
        return True
    except Exception:
        pass

    try:
        from PIL import Image

        imgs = [Image.fromarray(fr) for fr in normed]
        imgs[0].save(
            str(path),
            save_all=True,
            append_images=imgs[1:],
            duration=int(round(duration_s * 1000)),
            loop=int(loop),
        )
        return True
    except Exception:
        return False


@dataclass
class GifOptions:
    """Customization for optional rearrangement GIFs.

    Parameters
    ----------
    enabled
        When ``True``, accumulate round images and write GIF(s) under ``run_dir``.
    sources
        Which payloads to animate: ``"frame"``, ``"occupancy"``, or both.
    duration_s
        Seconds per GIF frame.
    loop
        GIF loop count (``0`` = infinite).
    max_side
        Optional downscale so ``max(H, W) <= max_side`` (keeps GIFs small).
    occupancy_cell_px
        Pixel size of each site in the occupancy heatmap used for GIFs/PNGs.
    auto_write
        Rewrite GIF files after every ``before_round`` dump (always up to
        date). If ``False``, call :meth:`SessionRecorder.finalize` once.
    """

    enabled: bool = True
    sources: Tuple[str, ...] = ("frame", "occupancy")
    duration_s: float = 0.4
    loop: int = 0
    max_side: Optional[int] = 512
    occupancy_cell_px: int = 16
    auto_write: bool = True


@dataclass
class VisualizationOptions:
    """Customization for optional per-round move-batch visualizations
    (schematic lattice view via ``aod_atommovr.imaging.visualization``; no
    camera-image rendering).

    Off by default: the recorder replays every move batch geometrically over
    ``RoundContext.occupancy`` (no live physics -- see this module's
    docstring) to render one panel per batch with move arrows, which isn't
    free. ``max_batches`` caps the panel count (uncapped, a round with many
    small parallel-move batches otherwise renders an unreadably tall
    figure); ``gif`` additionally renders an animated ``grid.gif`` cycling
    through the same snapshots. ``matplotlib`` is a soft dependency of this
    method only.
    """

    enabled: bool = False
    max_batches: Optional[int] = 15
    max_cols: int = 3
    grid_format: str = "svg"
    gif: bool = True
    gif_duration_s: float = 0.5
    gif_loop: int = 0


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


class SessionRecorder:
    """RoundHook: dump each round's frame/occupancy, append JSONL stats, and
    optionally render a schematic move-batch visualization.

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
    gif
        GIF customization. Pass ``GifOptions(enabled=False)`` to skip GIFs, or
        ``None`` for defaults (GIF on, frame+occupancy).
    visualization
        Move-batch visualization customization. ``None`` for defaults (off --
        pass ``VisualizationOptions(enabled=True, ...)`` to opt in).
    """

    def __init__(
        self,
        run_root: PathLike = "runs",
        *,
        enabled: bool = True,
        run_dir: Optional[PathLike] = None,
        meta: Optional[Mapping[str, Any]] = None,
        gif: Optional[GifOptions] = None,
        visualization: Optional[VisualizationOptions] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.run_root = run_root
        self._explicit_run_dir = Path(run_dir) if run_dir is not None else None
        self._meta = dict(meta) if meta else {}
        self.gif = gif if gif is not None else GifOptions()
        self.visualization = (
            visualization if visualization is not None else VisualizationOptions()
        )
        self.run_dir: Optional[Path] = None
        self._rounds_path: Optional[Path] = None
        self._gif_frames: Dict[str, List[np.ndarray]] = {
            "frame": [],
            "occupancy": [],
        }

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
            "gif": asdict(self.gif),
            "visualization": asdict(self.visualization),
        }
        payload.update(self._meta)
        (self.run_dir / "meta.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        self._rounds_path.touch(exist_ok=True)

    def before_round(self, ctx: RoundContext) -> None:
        """Dump frame + occupancy once per round (replaces the old
        acquire/detect stage folders -- a RoundContext only ever carries one
        already-measured frame + occupancy)."""
        if not self.enabled or self.run_dir is None:
            return

        round_dir = self.run_dir / f"round_{ctx.round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        if ctx.frame is not None:
            frame = np.asarray(ctx.frame)
            np.save(round_dir / "frame.npy", frame)
            _write_png(round_dir / "frame.png", frame)
            if self.gif.enabled and "frame" in self.gif.sources:
                self._gif_frames["frame"].append(
                    _resize_max_side(_to_uint8(frame), self.gif.max_side)
                )

        occ = np.asarray(ctx.occupancy)
        np.save(round_dir / "occupancy.npy", occ)
        heat = _occupancy_heatmap(occ, cell_px=self.gif.occupancy_cell_px)
        _write_png(round_dir / "occupancy.png", heat)
        if self.gif.enabled and "occupancy" in self.gif.sources:
            self._gif_frames["occupancy"].append(
                _resize_max_side(heat, self.gif.max_side)
            )

        if self.gif.enabled and self.gif.auto_write:
            self.write_gifs()

    def after_round(self, ctx: RoundContext) -> None:
        """Append one JSON line to ``rounds.jsonl``; optionally render a
        lattice GIF/SVG from ``ctx.occupancy`` + ``ctx.move_batches``."""
        if self.enabled and self._rounds_path is not None:
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

        self._save_move_visualization(ctx)

    def on_session_end(self, ctx: SessionContext) -> None:
        self.finalize()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _save_move_visualization(self, ctx: RoundContext) -> Optional[Path]:
        """Render ``ctx.move_batches`` as step-by-step lattice snapshots
        (initial occupancy + one panel per batch, with move arrows) via
        ``aod_atommovr.imaging.visualization`` -- schematic lattice view
        only (no camera-image rendering, no collision/failure detection:
        this replays *geometrically* over ``ctx.occupancy``, since a hook
        never sees the live, physics-aware ``AtomArray``).

        Always writes the static multi-panel ``grid.<grid_format>``; also
        writes an animated ``grid.gif`` cycling through the same snapshots
        when ``VisualizationOptions.gif`` is ``True`` (default).
        ``VisualizationOptions.max_batches`` caps how many batches get a
        panel/frame. No-ops unless both the recorder and
        ``self.visualization.enabled`` are true, or ``ctx.move_batches`` is
        empty. ``matplotlib`` is a soft dependency of this method only.
        Returns the round's visualization directory, or ``None`` if
        disabled/skipped.
        """
        if not self.enabled or self.run_dir is None or not self.visualization.enabled:
            return None
        if not ctx.move_batches:
            return None

        try:
            from aod_atommovr.imaging.visualization import (
                render_move_batch_frames,
                visualize_move_batches,
            )
        except ImportError:
            log.warning("matplotlib not available; skipping move visualization.")
            return None

        opts = self.visualization
        batches = list(ctx.move_batches)
        if opts.max_batches is not None and len(batches) > opts.max_batches:
            log.info(
                f"Round {ctx.round_idx}: move visualization truncated to "
                f"the first {opts.max_batches} of {len(batches)} batches "
                "(VisualizationOptions.max_batches)."
            )
            batches = batches[: opts.max_batches]

        stage_dir = self.run_dir / f"round_{ctx.round_idx:02d}_visualization"
        stage_dir.mkdir(parents=True, exist_ok=True)

        visualize_move_batches(
            ctx.occupancy,
            batches,
            save_path=str(stage_dir / f"grid.{opts.grid_format}"),
            title_suffix=f"round_{ctx.round_idx:02d}",
            max_cols=opts.max_cols,
        )

        if opts.gif:
            frames = render_move_batch_frames(ctx.occupancy, batches)
            _write_gif(
                stage_dir / "grid.gif",
                frames,
                duration_s=opts.gif_duration_s,
                loop=opts.gif_loop,
            )

        return stage_dir

    def write_gifs(self) -> dict[str, Path]:
        """Write accumulated GIF(s) under ``run_dir``. Returns written paths."""
        written: dict[str, Path] = {}
        if not self.enabled or self.run_dir is None or not self.gif.enabled:
            return written

        for source in self.gif.sources:
            frames = self._gif_frames.get(source) or []
            if len(frames) < 1:
                continue
            # frame → frames.gif; occupancy → occupancy.gif
            if source == "frame":
                out = self.run_dir / "frames.gif"
            elif source == "occupancy":
                out = self.run_dir / "occupancy.gif"
            else:
                out = self.run_dir / f"{source}.gif"

            if _write_gif(
                out,
                frames,
                duration_s=self.gif.duration_s,
                loop=self.gif.loop,
            ):
                written[source] = out
        return written

    def finalize(self) -> dict[str, Path]:
        """Flush GIFs (called at ``on_session_end``)."""
        return self.write_gifs()
