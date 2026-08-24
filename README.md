# atommovr-controller

## 0. Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Then:

```bash
git clone --recurse-submodules https://github.com/davidKMJ/atommovr-controller.git
cd atommovr-controller
uv sync
```

`--recurse-submodules` pulls `atommovr/`. If you already cloned without it:

```bash
git submodule update --init --recursive
```

`uv sync` is enough for algorithms, imaging, simulation, tests, and the notebook. Driving a real Spectrum card (Linux/NVIDIA) needs the extra plus the driver / IOMMU steps in [setup_guide.md](setup_guide.md):

```bash
uv sync --extra hardware
```

Open [`awg_controller.ipynb`](awg_controller.ipynb) (card / RF only) and [`atommovr_controller.ipynb`](atommovr_controller.ipynb) (settings → simulation → hardware). Kernel: `.venv`.

```bash
uv run pytest
```

## 1. Packages

The `atommovr/` submodule ([davidKMJ/atommovr](https://github.com/davidKMJ/atommovr)) is [bernienlab/atommovr](https://github.com/bernienlab/atommovr) with [SaBeBen/atommovr](https://github.com/SaBeBen/atommovr) merged in, plus timing edits. The main logic of `atommovr_controller/` and `awg_controller/` comes from that SaBeBen code, with significant feature additions and a refactor here.

**`atommovr/`** — rearrangement algorithms (Hungarian, PCFA, Tetris, BCv2, …) plus `AtomArray` / `Move` / `ErrorModel` / timing. See [atommovr/README.md](atommovr/README.md).

**`atommovr_controller/`** — control loop: camera → algorithm → RF. `engine=None` (the default) logs and sleeps; pass an `AWGEngine` to open the Spectrum card and play each round as one phase-continuous waveform (`stop` → `load_round` → `play`).

- `controller.py` — `AtommovrController`, `HardwareConfig`, `SoftwareConfig`, `configure_logging`
- `camera.py` — `Camera`, `OfflineArrayCamera` (synthetic), `RealArrayCamera` (hardware callback)
- `imaging/` — blob detect → rotate → grid assign; Gaussian PSF synthesis
- `hooks.py` — `RoundHook` observers on a read-only `RoundContext`

CLI and the notebook both call `configure_logging()` (stdout + `atommovr_controller.log`).

**`awg_controller/`** — `Move` → RF ramps → Spectrum card.

- `awg_control.py` — `RFConverter`, `AODSettings`. Summed tone amplitude ≤ 40% full-scale per channel
- `awg_engine.py` — ctypes to `native/libawg_engine.so`. Playback: `stream` (FIFO) or `memory` (full 1.25 GS/s)
- `native/` — CUDA/C engine via the vendored SCAPP SDK. Output voltage hard-capped at 2.0 V

**`recorder/`** — optional `RoundHook`: `meta.json` + one JSON line per round in `rounds.jsonl`. Attach with `AtommovrController(..., hooks=[Recorder(...)])`.

## 2. Structure

```
atommovr-controller/
├── atommovr_controller.ipynb       # settings, simulation, hardware
├── awg_controller.ipynb            # AWG / RFConverter / AWGEngine
├── setup_guide.md                  # NVIDIA driver + IOMMU for the Spectrum card
├── pyproject.toml
│
├── atommovr_controller/
│   ├── controller.py               # HardwareConfig, SoftwareConfig, AtommovrController
│   ├── camera.py                   # GaussianCameraConfig; RealArrayCamera(camera_fn=...)
│   ├── hooks.py                    # RoundHook / Hook / HookBus
│   ├── imaging/
│   │   ├── extraction.py           # BlobDetection, rotation, fit_grid_and_assign
│   │   ├── generation.py           # synthetic Gaussian frames for OfflineArrayCamera
│   │   ├── synthetic.py            # extraction scoring helpers (used by tests)
│   │   └── geometry.py
│   └── tests/
│
├── awg_controller/
│   ├── awg_control.py              # AODSettings; MAX_AMPLITUDE_PCT_PER_CHANNEL = 40
│   ├── awg_engine.py               # CardConfig, AWGEngineConfig, AWGEngine
│   ├── native/                     # CUDA/C SCAPP engine → libawg_engine.so
│   │   ├── Makefile
│   │   ├── awg_engine.h / .cu      # stream (FIFO) vs memory (card DRAM) playback
│   │   ├── phase.h / schedule.h    # phase-continuous ramps
│   │   ├── render.cuh              # GPU sample render
│   │   ├── stream.cuh              # FIFO RDMA
│   │   └── sequence.cuh            # memory-mode sequence replay
│   └── tests/
│
├── recorder/
│   └── recorder.py                 # run_root, enabled
│
├── atommovr/                       # submodule — https://github.com/davidKMJ/atommovr
│   └── atommovr/
│       ├── algorithms/             # Hungarian, PCFA, Tetris, BCv2, InsideOut, …
│       └── utils/                  # AtomArray, Move, ErrorModel, PhysicalParams, timing
│
└── scapp/                          # vendored Spectrum SDK
```
