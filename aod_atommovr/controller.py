#!/usr/bin/env python3
"""
atommovr Production Controller
================================
Orchestrates the complete atom-rearrangement feedback loop:

  1. Sync camera → occupancy (``Camera.sync``) — the camera is a required
     collaborator, not a hook (see ``hooks.py``'s docstring for why).
  2. ``hooks.before_round(ctx)`` — occupancy/frame already known.
  3. Check whether target is already filled → done.
  4. Compute rearrangement moves (configurable algorithm).
  5. Convert moves to RF ramps (RFConverter / AWGBatch).
  6. Write ramps to the Spectrum Instrumentation AWG card.
  7. Offline: advance physics on the controller ``AtomArray``.
  8. ``hooks.after_round(ctx)`` — always, including success/abort rounds.
  9. Repeat from step 1.

AWG generation onto a single card (multi-card support has been removed),
under a 40 % total-amplitude-per-channel safety budget.

Hardware backend note: the AWG real-time backend is being redesigned
around precomputing an entire round's waveform and streaming it via SCAPP
(see ``awg_controller.awg_engine.AWGEngine``); this controller isn't wired
up to it yet, so hardware init below is always simulation mode for now --
``_output_batch``/``_send_holding`` log what would have been sent and sleep
for the nominal duration instead of touching a card. An ``engine=`` may be
passed in and is stored, but is not (yet) driven by the loop.
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

from awg_controller.awg_control import AODSettings, AWGBatch, RFConverter
from awg_controller.awg_engine import AWGEngine

from aod_atommovr.camera import Camera, OfflineArrayCamera
from aod_atommovr.hooks import HookBus, RoundContext, RoundHook, SessionContext

#  No real-time hardware backend is wired up yet (see module docstring) --
#  always simulation mode.
_HW_AVAILABLE = False

#  logging
#  Handler setup only happens when this file is run as the entry point (see
#  `if __name__ == "__main__":` below) — importing this module as a library
#  (tests, notebooks, other scripts) must not have the side effect of
#  configuring the root logger or creating a log file.
log = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("atommovr_controller.log"),
        ],
    )


@dataclass
class HardwareConfig:
    """Spectrum Instrumentation card + AOD/physical configuration (mirrors
    cli.py defaults).

    A single card is opened -- multi-card support has been removed (it only
    ever broadcast identical commands to every card, with no real per-card
    partitioning).
    """

    #: Device path, e.g. "/dev/spcm0"
    card_path: str = "/dev/spcm0"

    #: Output amplitude - manufacturer maximum is 1.6 V into 50 Ω. Hard
    #: safety ceiling: must never exceed 2.0 V.
    max_amplitude_v: float = 1.6

    #: Output impedance
    output_load_ohms: float = 50.0

    #: AOD frequency-range and geometry.
    aod_settings: AODSettings = field(default_factory=AODSettings)

    #: Physical parameters (AOD speed, site spacing, loading probability …).
    physical_params: PhysicalParams = field(default_factory=PhysicalParams)


@dataclass
class SoftwareConfig:
    """Algorithm and imaging configuration."""

    #  Imaging
    #: Optional OpenCV SimpleBlobDetector_Params (or legacy dict — ignored by Camera).
    blob_params: Any = None

    #: Error model applied on the controller-owned ``AtomArray`` (offline physics).
    error_model: Optional[ErrorModel] = None

    #  Control loop
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


class AodController:
    """End-to-end atom rearrangement controller.

    Parameters
    ----------
    sw_config : SoftwareConfig
        Algorithm, imaging, and control-loop settings.
    hw_config : HardwareConfig
        Spectrum Instrumentation card settings, plus the AOD frequency/
        geometry (``aod_settings``) and physical parameters
        (``physical_params``) that describe the hardware being driven.
    camera : Camera
        Required collaborator (``OfflineArrayCamera`` or ``RealArrayCamera``)
        -- the round's occupancy source of truth. Not a hook; see
        ``hooks.py``'s docstring for why.
    engine : AWGEngine, optional
        Accepted and stored for future wiring; the loop does not drive it
        yet (see module docstring).
    hooks : Sequence[RoundHook], optional
        Observers fanned out to by a ``HookBus`` -- logging, recording,
        metrics. Bare callables are treated as ``after_round``-only.
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
        self.rf_converter = RFConverter(aod, hw_config.physical_params)

        err = (
            sw_config.error_model if sw_config.error_model is not None else ZeroNoise()
        )
        self.array = AtomArray(
            list(self.grid_shape),
            n_species=1,
            params=hw_config.physical_params,
            error_model=err,
        )

        self._grid_rotation: float = 0.0
        self._target_mask: Optional[np.ndarray] = None
        self._apply_target()

        self._initialize_hardware()

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    def _initialize_hardware(self) -> None:
        """No real-time hardware backend is wired up yet (see module
        docstring) -- always simulation mode."""
        log.info("Simulation mode: no hardware backend wired up.")

    def _output_batch(self, batch: AWGBatch) -> None:
        """Send one ``AWGBatch`` to the card."""
        log.info(
            f"[SIM] batch: {len(batch.ramps)} ramps, "
            f"duration={batch.travel_duration_s * 1e6:.1f} µs"
        )
        if batch.travel_duration_s > 0:
            time.sleep(batch.travel_duration_s)

    def _send_holding(self) -> None:
        """Restore static holding configuration (atoms held in place)."""
        holding = self.rf_converter.holding_config()
        log.info(f"[SIM] holding: {len(holding.ramps)} ramps")

    # ------------------------------------------------------------------
    # Imaging / targets
    # ------------------------------------------------------------------

    def _build_target_mask(self, grid_shape: Tuple[int, int]) -> np.ndarray:
        """Centred rectangular target from middle_size, falling back to
        AODSettings.target_rows/target_cols."""
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
        """Build / cache target mask and copy it onto ``self.array.target``."""
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
        """Execute the rearrangement feedback loop.

        Returns
        -------
        bool
            ``True`` if the target is successfully filled within ``max_rounds``.
        """
        cam = self.camera
        bus = self.hooks

        session_ctx = SessionContext(
            grid_shape=self.grid_shape,
            algorithm_name=self.sw.algorithm_name,
            max_rounds=self.sw.max_rounds,
        )
        bus.on_session_start(session_ctx)

        log.info(
            f"Loop start — algorithm={self.sw.algorithm_name}, "
            f"grid={self.grid_shape[0]}×{self.grid_shape[1]}, "
            f"max_rounds={self.sw.max_rounds}, camera={type(cam).__name__}"
        )

        try:
            for r in range(self.sw.max_rounds + 1):
                t_loop = time.perf_counter()

                # 1. Acquire + detect into the global array. The camera is
                # the round's occupancy source of truth, not a hook -- its
                # failures abort the round outright.
                try:
                    cam.sync(self.array)
                except Exception as exc:
                    log.error(f"Round {r}: acquisition failed - {exc}")
                    return False

                self._grid_rotation = float(getattr(cam, "grid_rotation", 0.0) or 0.0)
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

                # 2. Success?
                if filled == need:
                    log.info(f"SUCCESS - target filled after {r} round(s).")
                    ctx.success = True
                    bus.after_round(ctx)
                    return True

                if r == self.sw.max_rounds:
                    log.warning(
                        f"Max rounds ({self.sw.max_rounds}) reached; "
                        f"{need - filled} target sites remain empty."
                    )
                    ctx.success = False
                    bus.after_round(ctx)
                    break

                if atoms < need:
                    log.error(
                        f"Round {r}: insufficient atoms "
                        f"(have {atoms}, need {need}). Aborting."
                    )
                    ctx.aborted = "insufficient_atoms"
                    bus.after_round(ctx)
                    return False

                # 3. Algorithm
                try:
                    _, move_batches, algo_ok = self.algorithm.get_moves(self.array)
                except Exception as exc:
                    log.exception(f"Round {r}: algorithm raised - {exc}")
                    ctx.aborted = "algo_exception"
                    bus.after_round(ctx)
                    return False

                if not algo_ok:
                    log.error(f"Round {r}: algorithm reported failure.")
                    ctx.aborted = "algo_fail"
                    bus.after_round(ctx)
                    return False

                # 4. RF + hardware
                rf_batches = self.rf_converter.convert_sequence(move_batches)
                n_moves = sum(len(b) for b in move_batches)
                log.info(
                    f"Round {r}: {n_moves} moves → {len(rf_batches)} hardware batches."
                )

                for batch in rf_batches:
                    self._output_batch(batch)
                self._send_holding()

                # 5. Offline physics on the controller-owned array
                if isinstance(cam, OfflineArrayCamera):
                    self.array.evaluate_moves(move_batches)

                ctx.move_batches = move_batches
                ctx.rf_batches = rf_batches
                ctx.n_moves = n_moves
                ctx.total_travel_duration_s = float(
                    sum(b.travel_duration_s for b in rf_batches)
                )
                bus.after_round(ctx)

                elapsed_ms = (time.perf_counter() - t_loop) * 1e3
                log.info(f"Round {r} done in {elapsed_ms:.1f} ms.")

            return False
        finally:
            bus.on_session_end(session_ctx)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully stop the card and release all resources."""
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
        help="Card device path (single card; multi-card support removed)",
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

    with AodController(sw, hw, camera=camera) as ctrl:
        try:
            success = ctrl.run()
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
            sys.exit(130)


if __name__ == "__main__":
    _configure_logging()
    main()
