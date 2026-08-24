"""
AWG (Arbitrary Waveform Generator) control utilities for the atommovr pipeline.

Converts logical atom ``Move`` objects into RF ramp commands (``AWGBatch``)
executed on Spectrum Instrumentation cards via the SCAPP GPU-generation
backend (``awg_controller.scapp``).

Amplitude unit : percent of full-scale  (sum ≤ 40 % per channel, per manufacturer)
Frequency unit : Hz
Phase unit     : degrees
Time unit      : seconds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams
from atommovr.utils.timing import MIN_MOVE_DURATION_S, travel_duration_s

# ---------------------------------------------------------------------------
# Hardware constants  (source of truth: cli.py + spcm documentation)
# ---------------------------------------------------------------------------

#: Combined amplitude of all simultaneous tones on one output channel must
#: not exceed 40 % of full-scale (manufacturer recommendation).
MAX_AMPLITUDE_PCT_PER_CHANNEL: float = 40.0

#: Maximum sample rate of the Spectrum Instrumentation M4i.6631-x8 (16-bit,
#: 2-channel, PCIe x8 Gen2). ``AWGEngine.open()`` negotiates this unless
#: ``AWGEngineConfig.sample_rate_hz`` overrides it. Note: sustained
#: FIFO/SCAPP streaming over PCIe x8 Gen2 caps around ~700 MHz at
#: 16-bit/2-channel ("in excess of 2.8 GB/s" per the datasheet) -- below
#: this 1.25 GS/s onboard-replay maximum, so real ``sample_rate_hz``
#: choices should stay verified on real hardware.
M4I_6631_X8_MAX_SAMPLE_RATE_HZ: float = 1.25e9


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class AODSettings:
    """Frequency-range and geometry settings for the AWG / AOD pair.

    Site → RF mapping is index-linear over ``[f_min, f_max]`` (same as the
    imaging pipeline's grid indices after blob → rotate → assign).
    """

    # Frequency ranges (Hz)
    f_min_v: float = 60e6
    f_max_v: float = 100e6
    f_min_h: float = 60e6
    f_max_h: float = 100e6

    # Total trap-grid dimensions (DDS tone counts)
    grid_rows: int = 10
    grid_cols: int = 10

    # Target sub-array dimensions (informational, used by the controller)
    target_rows: int = 6
    target_cols: int = 6

    alignment: str = "center"  # "center" | "start"

    #: Approximate trap shift per MHz (µm/MHz).  Optics ballpark (~6.53 for
    #: the Zwierlein 808 nm / f1=75 / f2=400 / f_obj=28 setup); not used for RF.
    um_per_mhz: float = 6.526

    @property
    def f_spacing_v(self) -> float:
        """Row-axis inter-site frequency step (Hz)."""
        return (self.f_max_v - self.f_min_v) / max(self.grid_rows - 1, 1)

    @property
    def f_spacing_h(self) -> float:
        """Column-axis inter-site frequency step (Hz)."""
        return (self.f_max_h - self.f_min_h) / max(self.grid_cols - 1, 1)

    @property
    def fov_um_v(self) -> float:
        """Vertical field of view (µm) from RF bandwidth × ``um_per_mhz``."""
        return self.um_per_mhz * (self.f_max_v - self.f_min_v) / 1e6

    @property
    def fov_um_h(self) -> float:
        """Horizontal field of view (µm) from RF bandwidth × ``um_per_mhz``."""
        return self.um_per_mhz * (self.f_max_h - self.f_min_h) / 1e6


@dataclass
class RFRamp:
    """Single-tone RF command.

    Attributes
    ----------
    channel : int
        Hardware output channel (0 = V/row AOD, 1 = H/col AOD).
    core : int
        Mirrors ``tone_index``; no hardware meaning under SCAPP.
    f_start : float
        Pre-move frequency (Hz) — the current trap position.
    f_end : float
        Post-move frequency (Hz) — the endpoint of the GPU-synthesized ramp.
    amplitude_pct : float
        Per-tone amplitude (%).  All ramps on the same channel must sum
        to ≤ MAX_AMPLITUDE_PCT_PER_CHANNEL (40 %).
    phase_deg : float
        Tone phase offset (degrees), default 0.
    duration_s : float
        Travel duration (s) — Chebyshev × spacing / ``AOD_speed`` via
        ``atommovr.utils.timing.travel_duration_s`` — used as the GPU
        ramp-segment duration.
    tone_index : int
        Row index (channel 0) / column index (channel 1) this ramp
        addresses, always set by ``RFConverter``. Used to track per-tone
        phase continuity across GPU buffer fills. Default -1 only for
        ad-hoc ``RFRamp(...)`` constructions that don't care (e.g. tests).
    """

    channel: int
    core: int
    f_start: float
    f_end: float
    amplitude_pct: float
    phase_deg: float = 0.0
    duration_s: float = 0.0
    tone_index: int = -1


@dataclass
class AWGBatch:
    """Set of RF commands for one parallel move batch.

    ``travel_duration_s`` is the batch's travel window (Chebyshev × spacing /
    ``AOD_speed``), matching :func:`atommovr.utils.timing.travel_duration_s`.
    It is a single window, not a sum; totals across batches are named
    ``total_travel_duration_s``.
    """

    ramps: List[RFRamp]
    travel_duration_s: float


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class RFConverter:
    """Translate logical ``Move`` objects into ``AWGBatch`` hardware commands.

    Amplitude budget rule (from cli.py comments):
        Per-core amplitude = 40 % / n_simultaneous_tones_on_channel

    Move travel duration is computed by
    :func:`atommovr.utils.timing.travel_duration_s` (Chebyshev × spacing /
    ``AOD_speed``).  Parallel V and H tones overlap in time.

    Site frequencies are index-linear over ``AODSettings`` ``[f_min, f_max]``,
    matching the grid indices produced by imaging (blob → rotate → assign).

    Parameters
    ----------
    settings : AODSettings
    physical_params : PhysicalParams
    """

    def __init__(
        self,
        settings: AODSettings,
        physical_params: PhysicalParams,
    ) -> None:
        self.settings = settings
        self.params = physical_params
        # No DDS-core concept under SCAPP -- tones are software-summed
        # sines, so there's no fixed hardware tone-count ceiling. Always
        # sequential, uncapped.
        self._core_map = {
            0: list(range(settings.grid_rows)),
            1: list(range(settings.grid_cols)),
        }

    @property
    def core_map(self) -> Dict[int, List[int]]:
        """Per-channel tone index list (identical to ``RFRamp.tone_index``
        values), computed at construction time. No hardware meaning under
        SCAPP -- kept for indexing convenience.
        """
        return self._core_map

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_freq(self, row: int) -> float:
        return self.settings.f_min_v + row * self.settings.f_spacing_v

    def _col_to_freq(self, col: int) -> float:
        return self.settings.f_min_h + col * self.settings.f_spacing_h

    def _travel_duration_s(self, moves: List[Move]) -> float:
        """Travel duration (s) for the longest move in the batch."""
        return travel_duration_s(moves, self.params.spacing, self.params.AOD_speed)

    @staticmethod
    def _per_tone_amplitude(n: int) -> float:
        if n <= 0:
            return MAX_AMPLITUDE_PCT_PER_CHANNEL
        return MAX_AMPLITUDE_PCT_PER_CHANNEL / n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def holding_config(self) -> AWGBatch:
        """Static holding batch: every grid site at its resting frequency.

        All ``grid_rows`` V-cores and ``grid_cols`` H-cores are set with
        ``f_start == f_end`` (no motion).  Sent to the card between
        rearrangement rounds so atoms remain trapped.
        """
        v_cores = self._core_map[0]
        h_cores = self._core_map[1]
        amp_v = self._per_tone_amplitude(len(v_cores))
        amp_h = self._per_tone_amplitude(len(h_cores))

        ramps: List[RFRamp] = []
        for i, core in enumerate(v_cores):
            f = self._row_to_freq(i)
            ramps.append(
                RFRamp(
                    channel=0,
                    core=core,
                    f_start=f,
                    f_end=f,
                    amplitude_pct=amp_v,
                    tone_index=i,
                )
            )
        for j, core in enumerate(h_cores):
            f = self._col_to_freq(j)
            ramps.append(
                RFRamp(
                    channel=1,
                    core=core,
                    f_start=f,
                    f_end=f,
                    amplitude_pct=amp_h,
                    tone_index=j,
                )
            )
        return AWGBatch(ramps=ramps, travel_duration_s=0.0)

    def convert_moves(self, moves: List[Move]) -> AWGBatch:
        """Convert one parallel move batch into an ``AWGBatch``.

        Every grid row and column is included in the batch so that the
        total amplitude per channel stays constant at 40 %.  Rows and
        columns not involved in any move keep their resting frequency
        (``f_start == f_end``).

        The rearrangement algorithm guarantees that at most one target row
        is assigned per source row (and likewise for columns) within a
        single batch.

        Returns
        -------
        AWGBatch
            Holding batch if *moves* is empty, otherwise a full-grid batch
            with the moving tones updated.

        Raises
        ------
        ValueError
            If two moves assign conflicting targets to the same source
            row or column, or if a target index is out of bounds.
        """
        if not moves:
            return self.holding_config()

        duration_s = self._travel_duration_s(moves)
        v_cores = self._core_map[0]
        h_cores = self._core_map[1]
        amp_v = self._per_tone_amplitude(len(v_cores))
        amp_h = self._per_tone_amplitude(len(h_cores))

        # Build source → destination maps for each axis
        row_targets: Dict[int, int] = {}
        col_targets: Dict[int, int] = {}
        for m in moves:
            if m.from_row in row_targets and row_targets[m.from_row] != m.to_row:
                raise ValueError(
                    f"Conflicting row targets: row {m.from_row} → "
                    f"{row_targets[m.from_row]} and {m.to_row}"
                )
            if m.from_col in col_targets and col_targets[m.from_col] != m.to_col:
                raise ValueError(
                    f"Conflicting column targets: col {m.from_col} → "
                    f"{col_targets[m.from_col]} and {m.to_col}"
                )
            row_targets[m.from_row] = m.to_row
            col_targets[m.from_col] = m.to_col

        grid_rows = self.settings.grid_rows
        grid_cols = self.settings.grid_cols

        ramps: List[RFRamp] = []
        for row_idx, core in enumerate(v_cores):
            target_row = row_targets.get(row_idx, row_idx)
            if target_row < 0 or target_row >= grid_rows:
                raise ValueError(
                    f"Row move targets out-of-bounds index {target_row} "
                    f"(grid has {grid_rows} rows)."
                )
            ramps.append(
                RFRamp(
                    channel=0,
                    core=core,
                    f_start=self._row_to_freq(row_idx),
                    f_end=self._row_to_freq(target_row),
                    amplitude_pct=amp_v,
                    duration_s=duration_s,
                    tone_index=row_idx,
                )
            )
        for col_idx, core in enumerate(h_cores):
            target_col = col_targets.get(col_idx, col_idx)
            if target_col < 0 or target_col >= grid_cols:
                raise ValueError(
                    f"Column move targets out-of-bounds index {target_col} "
                    f"(grid has {grid_cols} columns)."
                )
            ramps.append(
                RFRamp(
                    channel=1,
                    core=core,
                    f_start=self._col_to_freq(col_idx),
                    f_end=self._col_to_freq(target_col),
                    amplitude_pct=amp_h,
                    duration_s=duration_s,
                    tone_index=col_idx,
                )
            )

        return AWGBatch(ramps=ramps, travel_duration_s=duration_s)

    def convert_sequence(self, move_batches: List[List[Move]]) -> List[AWGBatch]:
        """Convert a full rearrangement sequence into a list of ``AWGBatch`` objects."""
        return [self.convert_moves(b) for b in move_batches]
