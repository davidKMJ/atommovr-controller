"""Extract AOD calibration (frequency, power) points from a 1-D scan.

Oscilloscope/DAQ exports like ``data/data_example.txt`` are::

    Sample#, X-Step size = ..., Y-Step size = ... ;
    #1, <X>, <Y> ;

A frequency sweep of an ``n x n`` trap grid produces ``n`` regularly spaced
peaks on the X or Y channel. Those peak heights are the measured powers that
``amplitude_compensation.ipynb`` §2 feeds to ``AmplitudeCompensation.fit_*``.
Site *i* maps to frequency by the same index-linear rule as ``AODSettings``::

    f_i = f_min + i * (f_max - f_min) / (n - 1)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.signal import find_peaks, peak_prominences

DEFAULT_SCAN = Path("data/data_example.txt")
DEFAULT_F_MIN_HZ = 85e6
DEFAULT_F_MAX_HZ = 121e6
DEFAULT_N_SITES = 20


def load_scan(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` sample arrays from a ``Sample#, X, Y`` scan file."""
    xs: list[float] = []
    ys: list[float] = []
    with Path(path).open() as fh:
        next(fh)  # header
        for line in fh:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            xs.append(float(parts[1]))
            ys.append(float(parts[2]))
    if not xs:
        raise ValueError(f"no samples in {path}")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _baseline(x: np.ndarray) -> float:
    edge = max(8, len(x) // 10)
    return float(np.median(np.concatenate([x[:edge], x[-edge:]])))


def peak_intensities(x: np.ndarray, n_sites: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, baseline-subtracted heights)`` for ``n_sites`` peaks.

    Local maxima are ranked by prominence; the strongest ``n_sites`` are kept
    and returned in sample-index order (the order of the frequency sweep).
    """
    if n_sites < 2:
        raise ValueError("n_sites must be >= 2")
    min_distance = max(3, len(x) // (n_sites * 4))
    peaks, _ = find_peaks(x, distance=min_distance)
    if len(peaks) < n_sites:
        peaks, _ = find_peaks(x)
    if len(peaks) < n_sites:
        raise ValueError(
            f"found {len(peaks)} local maxima, need {n_sites}. "
            "Check the scan channel or n_sites."
        )
    prominences = peak_prominences(x, peaks)[0]
    keep = np.argsort(prominences)[-n_sites:]
    peaks = np.sort(peaks[keep])
    powers = x[peaks] - _baseline(x)
    if np.any(powers <= 0):
        raise ValueError(
            "baseline-subtracted peak heights must be positive; "
            "the scan may not contain a clear tone burst."
        )
    return peaks, powers


def calibration_from_scan(
    path: str | Path,
    channel: str,
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = DEFAULT_F_MAX_HZ,
    n_sites: int = DEFAULT_N_SITES,
    reverse: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Measured ``(CAL_FREQS_HZ, CAL_POWERS)`` for ``AmplitudeCompensation``.

    Parameters
    ----------
    path
        Scan file with ``Sample#, X, Y`` rows.
    channel
        ``"x"`` or ``"y"`` — which sample column holds the photodiode trace.
    f_min_hz, f_max_hz
        AOD band edges (Hz), matching ``AODSettings.f_min_*`` / ``f_max_*``.
    n_sites
        Tones along the swept axis (``grid_rows`` / ``grid_cols``).
    reverse
        Set if the sweep ran ``f_max → f_min`` instead of ``f_min → f_max``.
    """
    channel = channel.lower()
    if channel not in ("x", "y"):
        raise ValueError(f"channel must be 'x' or 'y', got {channel!r}")
    x, y = load_scan(path)
    trace = x if channel == "x" else y
    _, powers = peak_intensities(trace, n_sites)
    freqs = np.linspace(f_min_hz, f_max_hz, n_sites)
    if reverse:
        freqs = freqs[::-1]
    return freqs, powers


def format_notebook_block(freqs_hz: np.ndarray, powers: np.ndarray) -> str:
    """Copy-paste block for ``amplitude_compensation.ipynb`` §2."""
    freq_txt = ", ".join(f"{f:.6e}" for f in freqs_hz)
    power_txt = ", ".join(f"{p:.6g}" for p in powers)
    f_lo, f_hi = float(np.min(freqs_hz)), float(np.max(freqs_hz))
    return (
        f"F_LO, F_HI = {f_lo:.6e}, {f_hi:.6e}\n"
        f"CAL_FREQS_HZ = np.array([{freq_txt}])\n"
        f"CAL_POWERS = np.array([{power_txt}])\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", nargs="?", default=str(DEFAULT_SCAN), help="scan file")
    parser.add_argument("--f-min-mhz", type=float, default=DEFAULT_F_MIN_HZ / 1e6)
    parser.add_argument("--f-max-mhz", type=float, default=DEFAULT_F_MAX_HZ / 1e6)
    parser.add_argument("--n-sites", type=int, default=DEFAULT_N_SITES)
    parser.add_argument("--channel", choices=("x", "y"), default="x")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="sweep was f_max → f_min",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="optional path to write freqs_hz / powers arrays",
    )
    args = parser.parse_args()

    freqs, powers = calibration_from_scan(
        args.scan,
        args.channel,
        f_min_hz=args.f_min_mhz * 1e6,
        f_max_hz=args.f_max_mhz * 1e6,
        n_sites=args.n_sites,
        reverse=args.reverse,
    )
    print("# paste into amplitude_compensation.ipynb §2 Calibration data")
    print(format_notebook_block(freqs, powers))
    print("# site    f (MHz)    peak intensity")
    for i, (f, p) in enumerate(zip(freqs, powers)):
        print(f"# {i:4d}  {f / 1e6:8.4f}  {p:14.4f}")
    if args.npz is not None:
        args.npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.npz, freqs_hz=freqs, powers=powers)
        print(f"# wrote {args.npz}")


if __name__ == "__main__":
    main()
