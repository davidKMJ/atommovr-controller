"""
AWG (Arbitrary Waveform Generator) control utilities for the atommovr pipeline.

Amplitude unit : percent of full-scale  (sum ≤ 40 % per channel, per manufacturer)
Frequency unit : Hz
Phase unit     : degrees
Time unit      : seconds
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams
from atommovr.utils.timing import travel_duration_s

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------

#: Combined amplitude of all simultaneous tones on one output channel must
#: not exceed 40 % of full-scale (manufacturer recommendation).
MAX_AMPLITUDE_PCT_PER_CHANNEL: float = 40.0

#: Amplitude-compensation ratio equals 1.0 here.
REFERENCE_FREQUENCY_HZ: float = 100e6

_AMPLITUDE_MODE_STATIC: int = 0
_AMPLITUDE_MODE_LINEAR: int = 1
_AMPLITUDE_MODE_GAUSSIAN: int = 2
_AMPLITUDE_MODES: Dict[str, int] = {
    "linear": _AMPLITUDE_MODE_LINEAR,
    "gaussian": _AMPLITUDE_MODE_GAUSSIAN,
}


def _fit_linear(freqs_hz: np.ndarray, powers: np.ndarray) -> Tuple[float, float]:
    b, a = np.polyfit(freqs_hz, powers, 1)
    return float(a), float(b)


def _fit_gaussian(
    freqs_hz: np.ndarray, powers: np.ndarray, iters: int = 100
) -> Tuple[float, float, float, float]:
    """Levenberg-Marquardt in frequency units normalized to O(1) (raw Hz
    values make the Jacobian ill-conditioned).
    """
    f_mid = float(freqs_hz.mean())
    f_scale = float(freqs_hz.std()) or 1.0
    x = (freqs_hz - f_mid) / f_scale

    baseline = 0.5 * (float(powers[0]) + float(powers[-1]))
    idx = int(np.argmax(np.abs(powers - baseline)))
    x0 = float(x[idx])
    b = float(baseline - powers[idx]) or 1.0
    a = baseline
    s = max((x.max() - x.min()) / 6.0, 1e-3)
    p = np.array([a, b, x0, s], dtype=float)

    def residual(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b, x0, s = p
        d = x - x0
        e = np.exp(-(d**2) / (2.0 * s**2))
        return powers - (a - b * e), e, d

    resid, e, d = residual(p)
    cost = float(resid @ resid)
    lam = 1e-2
    for _ in range(iters):
        _, b, _, s = p
        jac = np.empty((len(x), 4))
        jac[:, 0] = 1.0
        jac[:, 1] = -e
        jac[:, 2] = -b * e * d / s**2
        jac[:, 3] = -b * e * d**2 / s**3
        jtj = jac.T @ jac
        jtr = jac.T @ resid
        diag = np.diag(jtj).copy()
        diag[diag == 0] = 1.0

        accepted = False
        for _ in range(30):
            try:
                delta = np.linalg.solve(jtj + lam * np.diag(diag), jtr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            p_new = p + delta
            p_new[3] = max(abs(p_new[3]), 1e-6)
            resid_new, e_new, d_new = residual(p_new)
            cost_new = float(resid_new @ resid_new)
            if cost_new < cost:
                p, resid, e, d, cost = p_new, resid_new, e_new, d_new, cost_new
                lam = max(lam / 10.0, 1e-12)
                accepted = True
                break
            lam *= 10.0
        if not accepted or np.linalg.norm(delta) < 1e-10 * (np.linalg.norm(p) + 1e-12):
            break

    a, b, x0, s = p
    return a, b, f_mid + x0 * f_scale, s * f_scale


@dataclass
class AmplitudeCompensation:
    """Amplitude-vs-frequency ratio, normalized to 1.0 at
    :data:`REFERENCE_FREQUENCY_HZ`. Per-tone amplitude is
    ``reference_amplitude_pct * this(f)`` (see ``RFConverter``).
    """

    mode: str  # "linear" | "gaussian"
    a: float
    b: float
    f0_hz: float = 0.0
    sigma_hz: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in _AMPLITUDE_MODES:
            raise ValueError(f"mode must be one of {sorted(_AMPLITUDE_MODES)}, got {self.mode!r}.")

    def __call__(self, f_hz: float) -> float:
        if self.mode == "linear":
            return self.a + self.b * f_hz
        d = f_hz - self.f0_hz
        return self.a - self.b * math.exp(-(d * d) / (2.0 * self.sigma_hz**2))

    @classmethod
    def fit_linear(cls, freqs_hz: Sequence[float], powers: Sequence[float]) -> "AmplitudeCompensation":
        """Regress measured (frequency, power) points to ``a + b*f``, then
        normalize so the fit equals 1.0 at ``REFERENCE_FREQUENCY_HZ``.
        """
        x = np.asarray(freqs_hz, dtype=float)
        y = np.asarray(powers, dtype=float)
        a, b = _fit_linear(x, y)
        ref = a + b * REFERENCE_FREQUENCY_HZ
        return cls("linear", a / ref, b / ref)

    @classmethod
    def fit_gaussian(cls, freqs_hz: Sequence[float], powers: Sequence[float]) -> "AmplitudeCompensation":
        """Regress measured (frequency, power) points to
        ``a - b*exp(-(f-f0)^2/(2*sigma^2))``, then normalize so the fit
        equals 1.0 at ``REFERENCE_FREQUENCY_HZ``.
        """
        x = np.asarray(freqs_hz, dtype=float)
        y = np.asarray(powers, dtype=float)
        a, b, f0, sigma = _fit_gaussian(x, y)
        ref = a - b * math.exp(-((REFERENCE_FREQUENCY_HZ - f0) ** 2) / (2.0 * sigma**2))
        return cls("gaussian", a / ref, b / ref, f0, sigma)


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
    amplitude_comp_mode, amplitude_comp_a/b/f0_hz/sigma_hz, amplitude_reference_pct :
        Native-engine amplitude-vs-frequency compensation (see
        ``AmplitudeCompensation``). Defaults (mode 0) mean amplitude_pct is
        held constant across the ramp -- the equal-power default.
    """

    channel: int
    f_start: float
    f_end: float
    amplitude_pct: float
    phase_deg: float = 0.0
    duration_s: float = 0.0
    tone_index: int = -1
    amplitude_comp_mode: int = _AMPLITUDE_MODE_STATIC
    amplitude_comp_a: float = 0.0
    amplitude_comp_b: float = 0.0
    amplitude_comp_f0_hz: float = 0.0
    amplitude_comp_sigma_hz: float = 0.0
    amplitude_reference_pct: float = 0.0


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
        Default (equal-power): per-tone amplitude = 40 % / n_simultaneous_tones_on_channel.
        With ``amplitude_compensation``: per-tone amplitude =
        ``reference_amplitude_pct * amplitude_compensation(f)``, not
        renormalized (raises if it exceeds the 40 % budget).

    Parameters
    ----------
    settings : AODSettings
    physical_params : PhysicalParams
    amplitude_compensation : AmplitudeCompensation, optional
        Non-default.
    reference_amplitude_pct : float, optional
        Required iff ``amplitude_compensation`` is set: the amplitude (%)
        at ``REFERENCE_FREQUENCY_HZ``, i.e. where the compensation ratio is 1.0.
    """

    def __init__(
        self,
        settings: AODSettings,
        physical_params: PhysicalParams,
        amplitude_compensation: Optional[AmplitudeCompensation] = None,
        reference_amplitude_pct: Optional[float] = None,
    ) -> None:
        if amplitude_compensation is not None and not (
            reference_amplitude_pct and reference_amplitude_pct > 0
        ):
            raise ValueError(
                "reference_amplitude_pct must be a positive % when amplitude_compensation is set."
            )
        self.settings = settings
        self.params = physical_params
        self.amplitude_compensation = amplitude_compensation
        self.reference_amplitude_pct = reference_amplitude_pct

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

    def _tone_amplitudes(self, freqs: List[float]) -> List[float]:
        """Per-tone amplitudes (%) for one channel.

        Default (``amplitude_compensation is None``): ``40 % / n``,
        independent of ``freqs``. Otherwise
        ``reference_amplitude_pct * amplitude_compensation(f)``, not
        renormalized -- only checked against the 40 % per-channel budget.
        """
        n = len(freqs)
        if n <= 0:
            return []
        if self.amplitude_compensation is None:
            return [MAX_AMPLITUDE_PCT_PER_CHANNEL / n] * n
        amps = [self.reference_amplitude_pct * self.amplitude_compensation(f) for f in freqs]
        if any(a <= 0 for a in amps):
            raise ValueError("amplitude_compensation must yield positive amplitudes.")
        total = sum(amps)
        if total > MAX_AMPLITUDE_PCT_PER_CHANNEL:
            raise ValueError(
                f"Compensated amplitudes sum to {total:.3f}%, exceeding the "
                f"{MAX_AMPLITUDE_PCT_PER_CHANNEL}% per-channel budget."
            )
        return amps

    def _ramp_kwargs(self) -> dict:
        """Native amplitude-compensation fields shared by every ramp in a batch."""
        if self.amplitude_compensation is None:
            return {}
        comp = self.amplitude_compensation
        return dict(
            amplitude_comp_mode=_AMPLITUDE_MODES[comp.mode],
            amplitude_comp_a=comp.a,
            amplitude_comp_b=comp.b,
            amplitude_comp_f0_hz=comp.f0_hz,
            amplitude_comp_sigma_hz=comp.sigma_hz,
            amplitude_reference_pct=self.reference_amplitude_pct,
        )

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
        freqs_v = [self._row_to_freq(i) for i in range(n_v)]
        freqs_h = [self._col_to_freq(j) for j in range(n_h)]
        amps_v = self._tone_amplitudes(freqs_v)
        amps_h = self._tone_amplitudes(freqs_h)
        comp_kwargs = self._ramp_kwargs()

        ramps: List[RFRamp] = []
        for i, f in enumerate(freqs_v):
            ramps.append(
                RFRamp(
                    channel=0,
                    f_start=f,
                    f_end=f,
                    amplitude_pct=amps_v[i],
                    tone_index=i,
                    **comp_kwargs,
                )
            )
        for j, f in enumerate(freqs_h):
            ramps.append(
                RFRamp(
                    channel=1,
                    f_start=f,
                    f_end=f,
                    amplitude_pct=amps_h[j],
                    tone_index=j,
                    **comp_kwargs,
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

        target_rows: List[int] = []
        for row_idx in range(n_v):
            target_row = row_targets.get(row_idx, row_idx)
            if target_row < 0 or target_row >= n_v:
                raise ValueError(
                    f"Row move targets out-of-bounds index {target_row} "
                    f"(grid has {n_v} rows)."
                )
            target_rows.append(target_row)
        target_cols: List[int] = []
        for col_idx in range(n_h):
            target_col = col_targets.get(col_idx, col_idx)
            if target_col < 0 or target_col >= n_h:
                raise ValueError(
                    f"Column move targets out-of-bounds index {target_col} "
                    f"(grid has {n_h} columns)."
                )
            target_cols.append(target_col)

        amps_v = self._tone_amplitudes([self._row_to_freq(t) for t in target_rows])
        amps_h = self._tone_amplitudes([self._col_to_freq(t) for t in target_cols])
        comp_kwargs = self._ramp_kwargs()

        ramps: List[RFRamp] = []
        for row_idx, target_row in enumerate(target_rows):
            ramps.append(
                RFRamp(
                    channel=0,
                    f_start=self._row_to_freq(row_idx),
                    f_end=self._row_to_freq(target_row),
                    amplitude_pct=amps_v[row_idx],
                    duration_s=duration_s,
                    tone_index=row_idx,
                    **comp_kwargs,
                )
            )
        for col_idx, target_col in enumerate(target_cols):
            ramps.append(
                RFRamp(
                    channel=1,
                    f_start=self._col_to_freq(col_idx),
                    f_end=self._col_to_freq(target_col),
                    amplitude_pct=amps_h[col_idx],
                    duration_s=duration_s,
                    tone_index=col_idx,
                    **comp_kwargs,
                )
            )

        return AWGBatch(ramps=ramps, travel_duration_s=duration_s)

    def convert_sequence(self, move_batches: List[List[Move]]) -> List[AWGBatch]:
        """Convert a full rearrangement sequence into a list of ``AWGBatch`` objects."""
        return [self.convert_moves(b) for b in move_batches]
