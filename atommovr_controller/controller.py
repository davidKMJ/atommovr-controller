#!/usr/bin/env python3
"""
Author: Claude Code, David Ko
Closed-loop atom rearrangement: camera → algorithm → AWG RF.

  1. Sync camera occupancy.
  2. ``hooks.before_round(ctx)``.
  3. Stop if the target is filled.
  4. Compute moves.
  5. Convert to RF ramps.
  6. Play on the AWG (or simulate).
  7. Offline: advance physics on ``AtomArray``.
  8. ``hooks.after_round(ctx)``.
  9. Repeat.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from atommovr.algorithms.single_species import (
    BCv2,
    BalanceAndCompact,
    GeneralizedBalance,
    Hungarian,
    ParallelHungarian,
    ParallelLBAP,
    PCFA,
    Tetris,
)
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.ErrorModel import ErrorModel
from atommovr.utils.core import Configurations, PhysicalParams
from atommovr.utils.errormodels import ZeroNoise

from awg_controller.awg_control import AmplitudeCompensation, AODSettings, AWGBatch, RFConverter
from awg_controller.awg_engine import AWGEngine, CardConfig

from atommovr_controller.camera import Camera, OfflineArrayCamera
from atommovr_controller.hooks import HookBus, RoundContext, RoundHook, SessionContext

# Library modules never attach handlers. CLI / notebook call configure_logging().
log = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
LOG_FILE = "atommovr_controller.log"


def configure_logging(*, filename: str = LOG_FILE) -> None:
    """Configure the root logger once (stdout + ``filename``).

    Safe to call from the CLI or the notebook. Importing this package
    does not configure logging, so tests and other libraries stay quiet
    until the caller sets handlers.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(filename),
        ],
    )


@dataclass
class HardwareConfig:
    """Card, AOD, and physical settings."""

    #: e.g. "/dev/spcm0"
    card_path: str = "/dev/spcm0"

    #: Manufacturer max is 1.6 V into 50 Ω; never exceed 2.0 V.
    max_amplitude_v: float = 1.6

    output_load_ohms: float = 50.0
    aod_settings: AODSettings = field(default_factory=AODSettings)
    physical_params: PhysicalParams = field(default_factory=PhysicalParams)

    #: Non-default. See ``RFConverter``/``AmplitudeCompensation``.
    amplitude_compensation_ch0: Optional[AmplitudeCompensation] = None
    amplitude_compensation_ch1: Optional[AmplitudeCompensation] = None
    #: Required iff either compensation is set.
    reference_amplitude_pct: Optional[float] = None


@dataclass
class SoftwareConfig:
    """Algorithm and imaging settings."""

    #: OpenCV blob-detector params (or a legacy dict — ignored by Camera).
    blob_params: Any = None
    #: Applied to the controller-owned ``AtomArray`` (offline physics).
    error_model: Optional[ErrorModel] = None

    max_rounds: int = 10
    algorithm_name: str = "PCFA"
    target_type: Configurations = Configurations.MIDDLE_FILL


_ALGORITHM_REGISTRY = {
    "PCFA": PCFA,
    "Hungarian": Hungarian,
    "Tetris": Tetris,
    "BalanceAndCompact": BalanceAndCompact,
    "BCv2": BCv2,
    "ParallelLBAP": ParallelLBAP,
    "ParallelHungarian": ParallelHungarian,
    "GeneralizedBalance": GeneralizedBalance,
}


class AtommovrController:
    """Camera → algorithm → AWG rearrangement loop.

    Parameters
    ----------
    sw_config : SoftwareConfig
        Algorithm, imaging, and loop settings.
    hw_config : HardwareConfig
        Card, AOD geometry, and physical parameters.
    camera : Camera
        Occupancy source of truth each round (not a hook).
    engine : AWGEngine, optional
        Opened at construction, closed on ``shutdown``. Default ``None``
        simulates with log + ``time.sleep``.
    hooks : Sequence[RoundHook], optional
        Observers (logging, recording, metrics). Bare callables run as
        ``after_round`` only.
    """

    def __init__(
        self,
        sw_config: SoftwareConfig,
        hw_config: HardwareConfig,
        camera: Camera,
        *,
        engine: Optional[AWGEngine] = None,
        hooks: Sequence[RoundHook] = (),
    ) -> None:
        self.sw = sw_config
        self.hw = hw_config
        self.engine = engine
        self.hooks = HookBus(hooks)

        aod = hw_config.aod_settings
        self.grid_shape: Tuple[int, int] = (int(aod.grid_rows), int(aod.grid_cols))

        if tuple(camera.grid_shape) != self.grid_shape:
            raise ValueError(
                f"camera.grid_shape {camera.grid_shape} != "
                f"AODSettings lattice {self.grid_shape}"
            )
        self.camera: Camera = camera

        self.algorithm = _ALGORITHM_REGISTRY[sw_config.algorithm_name]()
        self.rf_converter = RFConverter(
            aod,
            hw_config.physical_params,
            amplitude_compensation_ch0=hw_config.amplitude_compensation_ch0,
            amplitude_compensation_ch1=hw_config.amplitude_compensation_ch1,
            reference_amplitude_pct=hw_config.reference_amplitude_pct,
        )

        err = (
            sw_config.error_model if sw_config.error_model is not None else ZeroNoise()
        )
        self.array = AtomArray(
            list(self.grid_shape),
            n_species=1,
            params=hw_config.physical_params,
            error_model=err,
        )

        self._target_mask: Optional[np.ndarray] = None
        self._apply_target()

        self._initialize_hardware()

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    def _initialize_hardware(self) -> None:
        """Open the card, or log and return in simulation mode.

        Does not preload a holding round: ``load_round`` rejects
        zero-duration (all-static) batches.
        """
        if self.engine is None:
            log.info("[SIM] no AWGEngine attached.")
            return

        sample_rate_hz = self.engine.open()
        log.info("[HW] card opened: sample_rate=%.3f MS/s", sample_rate_hz / 1e6)

    def _play_round(self, rf_batches: Sequence[AWGBatch]) -> float:
        """Play one round as a single waveform, on hardware or simulated.
        Returns the round's total travel duration (s).

        Both modes treat the round the same way -- one unit, not one
        call per batch: compute the round's total travel duration once,
        then either drive the engine or just log, and sleep that total
        either way so the loop stays paced to real move time before the
        next ``camera.sync``.

        Hardware: ``stop()`` (required before ``load_round`` -- leaves a
        short RF gap, but keeps the previous tones parked until the last
        possible moment), ``load_round`` the whole round, ``play()``.
        """
        total_s = sum(b.travel_duration_s for b in rf_batches)
        if self.engine is None:
            log.info(
                "[SIM] round: %d batches, total travel=%.1f µs",
                len(rf_batches),
                total_s * 1e6,
            )
        else:
            self.engine.stop()
            self.engine.load_round(rf_batches)
            self.engine.play()
            log.info(
                "[HW] round: %d batches, total travel=%.1f µs",
                len(rf_batches),
                total_s * 1e6,
            )
        if total_s > 0:
            time.sleep(total_s)
        return total_s

    # ------------------------------------------------------------------
    # Imaging / targets
    # ------------------------------------------------------------------

    def _build_target_mask(self, grid_shape: Tuple[int, int]) -> np.ndarray:
        """Centred rectangular target from ``middle_size``, else AOD target size."""
        rows, cols = grid_shape
        ms = self.hw.physical_params.middle_size
        if ms is not None and len(ms) >= 2:
            tr, tc = int(ms[0]), int(ms[1])
        else:
            tr = int(self.hw.aod_settings.target_rows)
            tc = int(self.hw.aod_settings.target_cols)

        tr = min(max(tr, 1), rows)
        tc = min(max(tc, 1), cols)
        mask = np.zeros((rows, cols), dtype=int)
        r0 = max((rows - tr) // 2, 0)
        c0 = max((cols - tc) // 2, 0)
        mask[r0 : r0 + tr, c0 : c0 + tc] = 1
        return mask

    def _apply_target(self) -> np.ndarray:
        """Cache the target mask onto ``self.array.target``."""
        if self._target_mask is None:
            ms = self.hw.physical_params.middle_size
            if ms is not None and len(ms) >= 2:
                self.array.generate_target(self.sw.target_type, middle_size=list(ms))
                self._target_mask = (self.array.target[:, :, 0] > 0).astype(int)
            else:
                self._target_mask = self._build_target_mask(self.grid_shape)
                self.array.target[:, :, 0] = self._target_mask
        else:
            self.array.target[:, :, 0] = self._target_mask
        return self._target_mask

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """Run the loop. True if the target fills within ``max_rounds``."""
        cam = self.camera
        bus = self.hooks

        session_ctx = SessionContext(
            grid_shape=self.grid_shape,
            algorithm_name=self.sw.algorithm_name,
            max_rounds=self.sw.max_rounds,
        )
        bus.on_session_start(session_ctx)

        log.info(
            "Loop start: algorithm=%s, grid=%d×%d, max_rounds=%d, camera=%s",
            self.sw.algorithm_name,
            self.grid_shape[0],
            self.grid_shape[1],
            self.sw.max_rounds,
            type(cam).__name__,
        )

        try:
            for r in range(self.sw.max_rounds + 1):
                t_loop = time.perf_counter()

                # Occupancy. Camera failure aborts the round.
                try:
                    cam.sync(self.array)
                except Exception as exc:
                    log.error("Round %d: acquisition failed: %s", r, exc)
                    return False

                state = (self.array.matrix[:, :, 0] > 0).astype(int)
                target = self._apply_target()

                filled = int((state * target).sum())
                need = int(target.sum())
                atoms = int(state.sum())

                ctx = RoundContext(
                    round_idx=r,
                    occupancy=state.copy(),
                    target=target.copy(),
                    frame=getattr(cam, "last_frame", None),
                    atoms=atoms,
                    filled=filled,
                    need=need,
                )
                bus.before_round(ctx)

                if filled == need:
                    log.info("SUCCESS: target filled after %d round(s).", r)
                    ctx.success = True
                    bus.after_round(ctx)
                    return True

                if r == self.sw.max_rounds:
                    log.warning(
                        "Max rounds (%d) reached; %d target sites remain empty.",
                        self.sw.max_rounds,
                        need - filled,
                    )
                    ctx.success = False
                    bus.after_round(ctx)
                    break

                if atoms < need:
                    log.error(
                        "Round %d: insufficient atoms (have %d, need %d). Aborting.",
                        r,
                        atoms,
                        need,
                    )
                    ctx.aborted = "insufficient_atoms"
                    bus.after_round(ctx)
                    return False

                try:
                    _, move_batches, algo_ok = self.algorithm.get_moves(self.array)
                except Exception:
                    log.exception("Round %d: algorithm raised.", r)
                    ctx.aborted = "algo_exception"
                    bus.after_round(ctx)
                    return False

                if not algo_ok:
                    log.error("Round %d: algorithm reported failure.", r)
                    ctx.aborted = "algo_fail"
                    bus.after_round(ctx)
                    return False

                rf_batches = self.rf_converter.convert_sequence(move_batches)
                n_moves = sum(len(b) for b in move_batches)
                log.info(
                    "Round %d: %d moves → %d hardware batches.",
                    r,
                    n_moves,
                    len(rf_batches),
                )

                try:
                    total_travel_s = self._play_round(rf_batches)
                except Exception:
                    log.exception("Round %d: play failed.", r)
                    ctx.aborted = "play_fail"
                    bus.after_round(ctx)
                    return False

                if isinstance(cam, OfflineArrayCamera):
                    self.array.evaluate_moves(move_batches)

                ctx.move_batches = move_batches
                ctx.rf_batches = rf_batches
                ctx.n_moves = n_moves
                ctx.total_travel_duration_s = total_travel_s
                bus.after_round(ctx)

                elapsed_ms = (time.perf_counter() - t_loop) * 1e3
                log.info("Round %d done in %.1f ms.", r, elapsed_ms)

            return False
        finally:
            bus.on_session_end(session_ctx)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the card and release resources."""
        if self.engine is not None:
            self.engine.close()
            log.info("[HW] card closed.")
        log.info("Controller shut down.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="atommovr production controller — image → AOD feedback loop."
    )
    p.add_argument(
        "--algorithm",
        default="PCFA",
        choices=list(_ALGORITHM_REGISTRY),
        help="Rearrangement algorithm",
    )
    p.add_argument(
        "--grid-rows",
        type=int,
        default=10,
        help="Grid rows (V-AOD tones)",
    )
    p.add_argument("--grid-cols", type=int, default=5, help="Grid cols (H-AOD tones)")
    p.add_argument("--target-rows", type=int, default=6, help="Target sub-array rows")
    p.add_argument("--target-cols", type=int, default=5, help="Target sub-array cols")
    p.add_argument(
        "--max-rounds", type=int, default=10, help="Max rearrangement rounds"
    )
    p.add_argument(
        "--card",
        type=str,
        default="/dev/spcm0",
        help="Card device path",
    )
    p.add_argument(
        "--hardware",
        action="store_true",
        help="Drive a real AWGEngine instead of simulating.",
    )
    p.add_argument("--f-min-v", type=float, default=60e6, help="V-AOD f_min (Hz)")
    p.add_argument("--f-max-v", type=float, default=100e6, help="V-AOD f_max (Hz)")
    p.add_argument("--f-min-h", type=float, default=60e6, help="H-AOD f_min (Hz)")
    p.add_argument("--f-max-h", type=float, default=100e6, help="H-AOD f_max (Hz)")
    args = p.parse_args()

    hw = HardwareConfig(
        card_path=args.card,
        physical_params=PhysicalParams(
            middle_size=[args.target_rows, args.target_cols],
        ),
        aod_settings=AODSettings(
            f_min_v=args.f_min_v,
            f_max_v=args.f_max_v,
            f_min_h=args.f_min_h,
            f_max_h=args.f_max_h,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            target_rows=args.target_rows,
            target_cols=args.target_cols,
        ),
    )
    sw = SoftwareConfig(
        algorithm_name=args.algorithm,
        max_rounds=args.max_rounds,
    )
    camera = OfflineArrayCamera(
        (args.grid_rows, args.grid_cols),
        physical_params=hw.physical_params,
        blob_params=sw.blob_params,
    )

    engine = (
        AWGEngine(
            CardConfig(
                card_path=hw.card_path,
                max_amplitude_v=hw.max_amplitude_v,
                output_load_ohms=hw.output_load_ohms,
                aod_settings=hw.aod_settings,
            )
        )
        if args.hardware
        else None
    )

    with AtommovrController(sw, hw, camera=camera, engine=engine) as ctrl:
        try:
            success = ctrl.run()
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
            sys.exit(130)


if __name__ == "__main__":
    configure_logging()
    main()
