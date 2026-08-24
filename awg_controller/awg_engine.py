"""
Python control surface for the native SCAPP engine (``awg_controller/native``).

A round (a sequence of :class:`~awg_controller.awg_control.AWGBatch`) is
resolved into a phase-continuous two-channel waveform and played by the card.
There are two modes, differing only in how samples get there:

``"stream"``
    FIFO replay. Samples are rendered on the GPU just ahead of the card's read
    pointer, straight into the SCAPP RDMA ring. Round length is unbounded --
    memory is the ring alone -- but PCIe must sustain ``sample_rate * 4`` B/s
    forever, which an M4i's Gen2 x8 link cannot do much above 500 MS/s
    two-channel. Use for long, slow, watch-it-on-a-scope rounds.

``"memory"``
    Sequence replay from the card's own DRAM. The whole round is rendered up
    front and uploaded once, so there is no sustained streaming and the card
    runs at its full 1.25 GS/s. Round length is capped by
    ``dma_buffer_samples`` (see :meth:`AWGEngine.max_round_duration_s`). Use
    for real experiment rounds, which are milliseconds long.

Both modes park on the round's final frequencies indefinitely and
phase-exactly. Tone count per channel is fixed at construction
(``AODSettings.grid_rows``/``grid_cols``) and every batch must supply a ramp
for every tone, matching ``RFConverter``'s full-grid-every-batch invariant.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Sequence

from awg_controller.awg_control import AODSettings, AWGBatch

#: Hard safety ceiling, asserted before awg_engine_open is ever called.
MAX_SAFE_OUTPUT_V: float = 2.0

SHAPE_LINEAR = 0
SHAPE_SCURVE = 1

MODE_STREAM = 0
MODE_MEMORY = 1
_MODES = {"stream": MODE_STREAM, "memory": MODE_MEMORY}

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
NATIVE_DIR = os.path.abspath(os.path.join(MODULE_DIR, "native"))
LIB_PATH = os.path.join(NATIVE_DIR, "libawg_engine.so")


# ---------------------------------------------------------------------------
# ctypes <-> awg_engine.h binding
# ---------------------------------------------------------------------------


class _AWGEngineConfig(ctypes.Structure):
    """Mirrors ``AWGEngineConfig`` in awg_engine.h field-for-field."""

    _fields_ = [
        ("card_path", ctypes.c_char_p),
        ("max_amplitude_v", ctypes.c_double),
        ("output_load_ohms", ctypes.c_double),
        ("mode", ctypes.c_int32),
        ("notify_samples", ctypes.c_int32),
        ("dma_buffer_samples", ctypes.c_int64),
        ("fill_start_threshold_promille", ctypes.c_int32),
        ("hold_tail_samples", ctypes.c_int64),
        ("sample_rate_hz", ctypes.c_double),
        ("grid_rows", ctypes.c_int32),
        ("grid_cols", ctypes.c_int32),
        ("cuda_device_index", ctypes.c_int32),
    ]


class _AWGRoundRamp(ctypes.Structure):
    """Mirrors ``AWGRoundRamp`` in awg_engine.h field-for-field."""

    _fields_ = [
        ("channel", ctypes.c_int32),
        ("tone_index", ctypes.c_int32),
        ("f_start_hz", ctypes.c_double),
        ("f_end_hz", ctypes.c_double),
        ("amplitude_pct", ctypes.c_double),
        ("phase_deg", ctypes.c_double),
    ]


def _build_native_library() -> None:
    """Builds libawg_engine.so in-place via the Makefile next to the
    .cu source. Requires the CUDA toolkit (nvcc) and the SCAPP-capable spcm
    Linux driver library (-lspcm_linux) on the machine actually running
    this -- i.e. the machine attached to the card/GPU.
    """
    if not os.path.isdir(NATIVE_DIR):
        raise RuntimeError(
            f"awg_engine native source directory not found: {NATIVE_DIR}"
        )
    print(f"[INFO] Building {LIB_PATH} via `make` in {NATIVE_DIR} ...")
    subprocess.run(["make"], check=True, cwd=NATIVE_DIR)
    print("[INFO] Build completed.")


_lib_handle: Optional[ctypes.CDLL] = None


def _configure_signatures(lib: ctypes.CDLL) -> None:
    lib.awg_engine_open.argtypes = [ctypes.POINTER(_AWGEngineConfig)]
    lib.awg_engine_open.restype = ctypes.c_void_p

    lib.awg_engine_sample_rate_hz.argtypes = [ctypes.c_void_p]
    lib.awg_engine_sample_rate_hz.restype = ctypes.c_double

    lib.awg_engine_max_sample_value.argtypes = [ctypes.c_void_p]
    lib.awg_engine_max_sample_value.restype = ctypes.c_int16

    lib.awg_engine_max_round_samples.argtypes = [ctypes.c_void_p]
    lib.awg_engine_max_round_samples.restype = ctypes.c_int64

    lib.awg_engine_load_round.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),  # batch_travel_durations_s
        ctypes.c_int32,  # n_batches
        ctypes.POINTER(_AWGRoundRamp),  # ramps
        ctypes.c_int32,  # n_ramps
        ctypes.POINTER(ctypes.c_int32),  # batch_ramp_counts
        ctypes.c_int32,  # ramp_shape
    ]
    lib.awg_engine_load_round.restype = ctypes.c_int

    lib.awg_engine_total_travel_duration_s.argtypes = [ctypes.c_void_p]
    lib.awg_engine_total_travel_duration_s.restype = ctypes.c_double

    lib.awg_engine_play.argtypes = [ctypes.c_void_p]
    lib.awg_engine_play.restype = ctypes.c_int

    lib.awg_engine_last_error.argtypes = [ctypes.c_void_p]
    lib.awg_engine_last_error.restype = ctypes.c_char_p

    lib.awg_engine_stop.argtypes = [ctypes.c_void_p]
    lib.awg_engine_stop.restype = None

    lib.awg_engine_close.argtypes = [ctypes.c_void_p]
    lib.awg_engine_close.restype = None


def _lib() -> ctypes.CDLL:
    """Loads libawg_engine.so, building it once if missing (same
    build-on-demand shape as
    ``atommovr.algorithms.source.PPSU_weight_matching.load_shared_library``,
    adapted for an nvcc-built Makefile target rather than a setuptools
    Extension).
    """
    global _lib_handle
    if _lib_handle is not None:
        return _lib_handle
    if not os.path.isfile(LIB_PATH):
        _build_native_library()
    try:
        lib = ctypes.CDLL(LIB_PATH)
    except OSError:
        _build_native_library()
        lib = ctypes.CDLL(LIB_PATH)
    _configure_signatures(lib)
    _lib_handle = lib
    return lib


# ---------------------------------------------------------------------------
# High-level Python API
# ---------------------------------------------------------------------------


@dataclass
class CardConfig:
    """Spectrum Instrumentation card + AOD settings that :class:`AWGEngine`
    opens the card with. Kept separate from any controller-level hardware
    config so this module has no dependency on ``aod_atommovr``.
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


@dataclass
class AWGEngineConfig:
    """Tuning knobs for :class:`AWGEngine`.

    Defaults target ``"stream"`` mode. For ``"memory"`` mode raise
    ``sample_rate_hz`` to the card's maximum and size ``dma_buffer_samples``
    to the longest round you need -- see the field notes below.
    """

    #: "stream" (FIFO, unbounded length, rate-limited) or "memory" (sequence
    #: replay from card DRAM, full rate, bounded length).
    mode: str = "stream"
    #: STREAM only: frames rendered per DMA notification. WAITDMA wakes once
    #: per chunk, so notify_samples/sample_rate is the render budget; the
    #: ring absorbs jitter but not the average rate. 262144 gives ~210 us of
    #: headroom at 1.25 GS/s -- smaller chunks (e.g. 16384, ~13 us) underrun.
    notify_samples: int = 262144
    #: STREAM: ring depth / jitter budget. MEMORY: upload staging buffer,
    #: i.e. the maximum round length. Either way this is pinned for GPUDirect
    #: RDMA and must fit the GPU's BAR1 aperture (`nvidia-smi -q | grep -A3
    #: "BAR1 Memory Usage"`; frames are 4 bytes -- a 256 MB BAR1 tops out
    #: near 48M frames).
    dma_buffer_samples: int = 16 * 1024 * 1024
    fill_start_threshold_promille: int = 800
    #: MEMORY only: length of the looped park segment the card rests on
    #: after the round. Must be a power of two so it holds a whole number of
    #: cycles of every tone (frequencies snap to the resulting
    #: sample_rate/hold_tail_samples grid).
    hold_tail_samples: int = 1 << 20
    #: None -> the card's maximum. Safe in MEMORY mode. In STREAM mode, an
    #: M4i.6631-x8's Gen2 x8 PCIe link sustains roughly 500-800 MS/s
    #: two-channel (well under its 1.25 GS/s onboard-replay maximum) --
    #: raise only against a measured link rate; exceeding it underruns
    #: partway into the round rather than failing at startup.
    sample_rate_hz: Optional[float] = 500e6
    #: Frequency-ramp shape for non-static moves ("linear" | "scurve").
    ramp_shape: str = "linear"
    cuda_device_index: int = 0

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(
                f"mode must be one of {sorted(_MODES)}, got {self.mode!r}."
            )
        if self.dma_buffer_samples <= 0:
            raise ValueError("dma_buffer_samples must be positive.")
        if self.mode == "memory":
            tail = self.hold_tail_samples
            if tail <= 0 or tail & (tail - 1):
                raise ValueError(
                    f"hold_tail_samples must be a power of two, got {tail}."
                )
            return
        if self.notify_samples <= 0:
            raise ValueError("notify_samples must be positive.")
        if self.dma_buffer_samples % self.notify_samples != 0:
            raise ValueError(
                f"dma_buffer_samples ({self.dma_buffer_samples}) must be an "
                f"exact multiple of notify_samples ({self.notify_samples})."
            )


class AWGEngine:
    """Life-cycle::

    engine = AWGEngine(card, config)   # card.aod_settings describes the AOD
    engine.open()                # opens card+GPU, negotiates sample rate
    engine.load_round(batches)   # resolves the round into a GPU segment schedule
    engine.play()                # blocks until card.start() fires; renders as it goes
    engine.stop()                # halts playback; load_round()+play() for the next round
    engine.close()               # releases the card/GPU
    """

    def __init__(
        self,
        card: CardConfig,
        config: Optional[AWGEngineConfig] = None,
    ) -> None:
        self._card = card
        self._aod = card.aod_settings
        self._config = config or AWGEngineConfig()
        self._handle: Optional[int] = None

    def open(self) -> float:
        """Opens the card + CUDA device and negotiates the sample rate.
        Returns the negotiated rate (Hz).
        """
        if self._card.max_amplitude_v > MAX_SAFE_OUTPUT_V:
            raise ValueError(
                f"max_amplitude_v={self._card.max_amplitude_v} V exceeds "
                f"{MAX_SAFE_OUTPUT_V} V hard safety ceiling."
            )

        lib = _lib()
        cfg = _AWGEngineConfig(
            card_path=self._card.card_path.encode(),
            max_amplitude_v=self._card.max_amplitude_v,
            output_load_ohms=self._card.output_load_ohms,
            mode=_MODES[self._config.mode],
            notify_samples=self._config.notify_samples,
            dma_buffer_samples=self._config.dma_buffer_samples,
            fill_start_threshold_promille=self._config.fill_start_threshold_promille,
            hold_tail_samples=self._config.hold_tail_samples,
            sample_rate_hz=self._config.sample_rate_hz or 0.0,
            grid_rows=self._aod.grid_rows,
            grid_cols=self._aod.grid_cols,
            cuda_device_index=self._config.cuda_device_index,
        )
        handle = lib.awg_engine_open(ctypes.byref(cfg))
        if not handle:
            raise RuntimeError(
                "awg_engine_open failed -- see stderr above for the native error message."
            )
        self._handle = handle

        sample_rate_hz = lib.awg_engine_sample_rate_hz(handle)
        max_tone_freq = max(self._aod.f_max_v, self._aod.f_max_h)
        if max_tone_freq >= 0.5 * sample_rate_hz:
            lib.awg_engine_close(handle)
            self._handle = None
            raise ValueError(
                f"Tone frequency {max_tone_freq / 1e6:.1f} MHz exceeds Nyquist "
                f"({sample_rate_hz / 2e6:.1f} MHz) at sample_rate="
                f"{sample_rate_hz / 1e6:.1f} MHz."
            )
        return sample_rate_hz

    def load_round(self, batches: Sequence[AWGBatch]) -> None:
        """Resolve *batches* (a full round, played back-to-back) into a
        segment schedule, replacing any previously loaded round.

        In ``"stream"`` mode this builds only the schedule (~2 MB for 500
        batches x 60 tones) and is cheap; samples are rendered during
        :meth:`play`. In ``"memory"`` mode it also renders the whole round and
        uploads it to the card, so this is the expensive call and it will
        raise if the round exceeds :attr:`max_round_duration_s`.

        Every batch must supply a ramp for every tone (``grid_rows +
        grid_cols`` ramps), and the first batch's ``f_start`` per tone is
        treated as the pre-existing resting frequency -- matching
        ``awg_controller.scapp.synthesize_round_waveform``.

        Once the round is exhausted the engine parks on the final
        frequencies, indefinitely and phase-exactly, until :meth:`stop`.
        """
        self._require_open()
        lib = _lib()

        n_batches = len(batches)
        travel_durations_s = (ctypes.c_double * n_batches)(
            *(b.travel_duration_s for b in batches)
        )
        counts = (ctypes.c_int32 * n_batches)(*(len(b.ramps) for b in batches))

        n_ramps = sum(len(b.ramps) for b in batches)
        ramps = (_AWGRoundRamp * n_ramps)()
        i = 0
        for batch in batches:
            for ramp in batch.ramps:
                ramps[i] = _AWGRoundRamp(
                    channel=ramp.channel,
                    tone_index=ramp.tone_index,
                    f_start_hz=ramp.f_start,
                    f_end_hz=ramp.f_end,
                    amplitude_pct=ramp.amplitude_pct,
                    phase_deg=ramp.phase_deg,
                )
                i += 1

        shape = SHAPE_SCURVE if self._config.ramp_shape == "scurve" else SHAPE_LINEAR
        rc = lib.awg_engine_load_round(
            self._handle, travel_durations_s, n_batches, ramps, n_ramps, counts, shape
        )
        if rc != 0:
            raise RuntimeError(f"awg_engine_load_round failed: {self.last_error}")

    @property
    def look_ahead_s(self) -> float:
        """STREAM: seconds of waveform the ring holds -- the jitter budget.
        A stall longer than this underruns.
        """
        self._require_open()
        return self._config.dma_buffer_samples / self.sample_rate_hz

    @property
    def max_round_duration_s(self) -> float:
        """Longest round this engine will accept.

        ``inf`` in ``"stream"`` mode. In ``"memory"`` mode the round must fit
        the RDMA staging buffer, so this is
        ``dma_buffer_samples / sample_rate`` -- e.g. 32M frames (134 MB, safe
        against a 256 MB BAR1) is 26.8 ms at 1.25 GS/s.
        """
        self._require_open()
        limit = _lib().awg_engine_max_round_samples(self._handle)
        if limit >= 2**62:
            return float("inf")
        return limit / self.sample_rate_hz

    @property
    def total_travel_duration_s(self) -> float:
        """Summed travel window (s) over every batch of the loaded round."""
        self._require_open()
        return _lib().awg_engine_total_travel_duration_s(self._handle)

    def play(self) -> None:
        """Starts playback of the loaded round.

        STREAM blocks until the ring has pre-filled and the card has started;
        MEMORY returns as soon as the card is triggered.
        """
        self._require_open()
        rc = _lib().awg_engine_play(self._handle)
        if rc != 0:
            raise RuntimeError(f"awg_engine_play failed: {self.last_error}")

    @property
    def sample_rate_hz(self) -> float:
        self._require_open()
        return _lib().awg_engine_sample_rate_hz(self._handle)

    @property
    def max_sample_value(self) -> int:
        self._require_open()
        return int(_lib().awg_engine_max_sample_value(self._handle))

    @property
    def last_error(self) -> Optional[str]:
        if self._handle is None:
            return None
        msg = _lib().awg_engine_last_error(self._handle)
        return msg.decode() if msg else None

    def stop(self) -> None:
        """Halts playback (if running). The card stays open -- load_round()
        and play() again without reopening.
        """
        if self._handle is not None:
            _lib().awg_engine_stop(self._handle)

    def close(self) -> None:
        """Stops (if needed) and releases the card + GPU resources."""
        if self._handle is not None:
            _lib().awg_engine_close(self._handle)
            self._handle = None

    def _require_open(self) -> None:
        if self._handle is None:
            raise RuntimeError("call open() first")

    def __enter__(self) -> "AWGEngine":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
