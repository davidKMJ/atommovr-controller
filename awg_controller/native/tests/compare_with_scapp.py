"""Diff the native fixed-point schedule against the Python reference.

Runs on a laptop: no CUDA, no card. Builds one round, renders it two ways --
`awg_controller.scapp.synthesize_round_waveform` (float64, the reference)
and the native header-only schedule (Q0.64 fixed point) -- and compares sample
by sample.

    c++ -O2 -std=c++14 -o /tmp/test_schedule awg_controller/native/tests/test_schedule.cpp
    python awg_controller/native/tests/compare_with_scapp.py /tmp/test_schedule
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Fallback for running this script directly (not via an installed/editable
# `awg_controller`): repo root on sys.path.
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from awg_controller.awg_control import AWGBatch, RFRamp  # noqa: E402
from awg_controller.scapp import synthesize_round_waveform  # noqa: E402

FS = 10e6
N_V, N_H = 2, 1
SHAPE_LINEAR, SHAPE_SCURVE = 0, 1

# (duration_s, [(channel, tone_index, f_start, f_end, amplitude_pct, phase_deg)])
# A nonzero phase_deg is deliberate: the static offset must NOT be folded into
# the batch-to-batch carry, and only a nonzero value exposes that bug.
ROUND = [
    (
        20e-6,
        [
            (0, 0, 1.0e6, 1.0e6, 30.0, 0.0),
            (0, 1, 1.2e6, 1.2e6, 25.0, 37.0),
            (1, 0, 2.0e6, 2.0e6, 40.0, 0.0),
        ],
    ),
    (
        50e-6,
        [
            (0, 0, 1.0e6, 1.3e6, 30.0, 0.0),
            (0, 1, 1.2e6, 1.1e6, 25.0, 37.0),
            (1, 0, 2.0e6, 2.5e6, 40.0, 12.5),
        ],
    ),
    (
        0.0,
        [
            (0, 0, 1.3e6, 1.3e6, 30.0, 0.0),  # zero-duration: state only
            (0, 1, 1.1e6, 1.1e6, 25.0, 37.0),
            (1, 0, 2.5e6, 2.5e6, 40.0, 12.5),
        ],
    ),
    (
        30e-6,
        [
            (0, 0, 1.3e6, 1.3e6, 30.0, 0.0),
            (0, 1, 1.1e6, 1.1e6, 25.0, 37.0),
            (1, 0, 2.5e6, 2.5e6, 40.0, 12.5),
        ],
    ),
]


def write_round_file(path: Path, shape: int) -> None:
    with path.open("w") as fh:
        fh.write(f"{FS!r} {N_V} {N_H} {shape} {len(ROUND)}\n")
        for dur, ramps in ROUND:
            fh.write(f"{dur!r} {len(ramps)}\n")
            for ch, ti, f0, f1, amp, deg in ramps:
                fh.write(f"{ch} {ti} {f0!r} {f1!r} {amp!r} {deg!r}\n")


def python_reference(shape: int) -> dict[int, np.ndarray]:
    batches = [
        AWGBatch(
            ramps=[
                RFRamp(
                    channel=ch,
                    core=ti,
                    f_start=f0,
                    f_end=f1,
                    amplitude_pct=amp,
                    phase_deg=deg,
                    tone_index=ti,
                )
                for ch, ti, f0, f1, amp, deg in ramps
            ],
            travel_duration_s=dur,
        )
        for dur, ramps in ROUND
    ]
    return synthesize_round_waveform(
        batches, FS, ramp_shape="scurve" if shape == SHAPE_SCURVE else "linear"
    )


def native(binary: str, round_file: Path) -> dict[int, np.ndarray]:
    out = subprocess.run([binary, str(round_file)], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"native renderer failed:\n{out.stderr}")
    per_ch: dict[int, list[float]] = {}
    for line in out.stdout.splitlines():
        ch_s, _s, v_s = line.split()
        per_ch.setdefault(int(ch_s), []).append(float(v_s))
    return {ch: np.asarray(v) for ch, v in per_ch.items()}


def main() -> int:
    binary = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_schedule"
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for shape, label in ((SHAPE_LINEAR, "linear"), (SHAPE_SCURVE, "scurve")):
            rf = Path(td) / f"round_{label}.txt"
            write_round_file(rf, shape)
            ref = python_reference(shape)
            got = native(binary, rf)

            print(f"\n=== ramp_shape = {label} ===")
            for ch in sorted(ref):
                a, b = ref[ch], got.get(ch)
                if b is None or a.shape != b.shape:
                    print(
                        f"  ch{ch}: SHAPE MISMATCH ref={a.shape} native="
                        f"{None if b is None else b.shape}"
                    )
                    bad += 1
                    continue
                err = float(np.max(np.abs(a - b)))
                # amplitudes sum to <=1.0, so express the error in int16 codes
                ok = err < 1e-6
                print(
                    f"  ch{ch}: n={a.size:5d}  max|diff|={err:.3e}"
                    f"  ({err * 32767:7.4f} LSB16)  {'PASS' if ok else 'FAIL'}"
                )
                bad += 0 if ok else 1

    print("\n" + ("FAILURES PRESENT" if bad else "native schedule matches scapp.py"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
