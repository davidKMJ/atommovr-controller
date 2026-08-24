"""
AWG (Arbitrary Waveform Generator) control utilities for the atommovr pipeline.

Amplitude unit : percent of full-scale  (sum ≤ 40 % per channel, per manufacturer)
Frequency unit : Hz
Phase unit     : degrees
Time unit      : seconds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams
from atommovr.utils.timing import travel_duration_s

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------

#: Combined amplitude of all simultaneous tones on one output channel must
#: not exceed 40 % of full-scale (manufacturer recommendation).
MAX_AMPLITUDE_PCT_PER_CHANNEL: float = 40.0


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
    f_min_v: float = 85e6
    f_max_v: float = 121e6
    f_min_h: float = 85e6
    f_max_h: float = 121e6

    # Total trap-grid dimensions
    grid_rows: int = 10
    grid_cols: int = 10

    # Target sub-array dimensions
    target_rows: int = 6
    target_cols: int = 6

    alignment: str = "center"  # "center" | "start"

    #: Approximate trap shift per MHz (µm/MHz)
    um_per_mhz: Optional[float] = None

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
        Travel duration (s) — Chebyshev × spacing / ``AOD_speed``
    tone_index : int
        Row index (channel 0) / column index (channel 1) this ramp
        addresses, always set by ``RFConverter``. Default -1 only for
        ad-hoc ``RFRamp(...)`` constructions that don't care (e.g. tests).
    """

    channel: int
    f_start: float
    f_end: float
    amplitude_pct: float
    phase_deg: float = 0.0
    duration_s: float = 0.0
    tone_index: int = -1


@dataclass
class AWGBatch:
    """Set of RF commands for one parallel move batch."""

    ramps: List[RFRamp]
    travel_duration_s: float


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class RFConverter:
    """Translate logical ``Move`` objects into ``AWGBatch`` hardware commands.

    Amplitude budget rule:
        Per-tone amplitude = 40 % / n_simultaneous_tones_on_channel

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

        All ``grid_rows`` V-tones and ``grid_cols`` H-tones are set with
        ``f_start == f_end`` (no motion).  Sent to the card between
        rearrangement rounds so atoms remain trapped.
        """
        n_v = self.settings.grid_rows
        n_h = self.settings.grid_cols
        amp_v = self._per_tone_amplitude(n_v)
        amp_h = self._per_tone_amplitude(n_h)

        ramps: List[RFRamp] = []
        for i in range(n_v):
            f = self._row_to_freq(i)
            ramps.append(
                RFRamp(
                    channel=0,
                    f_start=f,
                    f_end=f,
                    amplitude_pct=amp_v,
                    tone_index=i,
                )
            )
        for j in range(n_h):
            f = self._col_to_freq(j)
            ramps.append(
                RFRamp(
                    channel=1,
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
        n_v = self.settings.grid_rows
        n_h = self.settings.grid_cols
        amp_v = self._per_tone_amplitude(n_v)
        amp_h = self._per_tone_amplitude(n_h)

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

        ramps: List[RFRamp] = []
        for row_idx in range(n_v):
            target_row = row_targets.get(row_idx, row_idx)
            if target_row < 0 or target_row >= n_v:
                raise ValueError(
                    f"Row move targets out-of-bounds index {target_row} "
                    f"(grid has {n_v} rows)."
                )
            ramps.append(
                RFRamp(
                    channel=0,
                    f_start=self._row_to_freq(row_idx),
                    f_end=self._row_to_freq(target_row),
                    amplitude_pct=amp_v,
                    duration_s=duration_s,
                    tone_index=row_idx,
                )
            )
        for col_idx in range(n_h):
            target_col = col_targets.get(col_idx, col_idx)
            if target_col < 0 or target_col >= n_h:
                raise ValueError(
                    f"Column move targets out-of-bounds index {target_col} "
                    f"(grid has {n_h} columns)."
                )
            ramps.append(
                RFRamp(
                    channel=1,
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
