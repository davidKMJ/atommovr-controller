"""
SCAPP phase/waveform math for the Spectrum Instrumentation AWG card.
=====================================================================

Pure, hardware-free math shared by the precompute engine
(``awg_controller/native/awg_engine``, C/CUDA) and offline tooling
(``SessionRecorder.save_spectrogram``, spectrogram/trajectory plots): the
closed-form per-tone phase/frequency formulas for a "hold" (constant),
"linear", or "scurve" frequency ramp, and the phase-continuity carry applied
at every transition between them.

The precompute engine's CUDA kernel evaluates this same math directly in C
(see ``awg_engine.cu``'s ``instantaneous_phase``) rather than
importing this module -- kept in sync by construction since both are
transcriptions of the identical formulas, not by sharing code across the
Python/CUDA boundary.

Safety
------
* Maximum output voltage MUST stay below 2.0 V in all scripts
  (``awg_controller.awg_engine.MAX_SAFE_OUTPUT_V``).
* Per-channel tone amplitudes are normalised so their sum never exceeds
  ``MAX_AMPLITUDE_PCT_PER_CHANNEL`` (40 %), bounding the digital sum to
  well within full scale regardless of tone count or phase alignment.
* Always verify amplifier output with an oscilloscope before connecting
  to the AOD. Excessive voltage will damage the AOD driver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from awg_controller.awg_control import AWGBatch

TWO_PI: float = 2.0 * math.pi


@dataclass(frozen=True)
class ToneSegment:
    """Closed-form frequency/phase trajectory for one tone on one channel.

    ``shape`` selects the frequency profile over ``[0, duration_s]``:

    * ``"hold"`` — constant at ``f_end`` (``duration_s`` is informational,
      not used in the phase formula).
    * ``"linear"`` — constant-slope sweep from ``f_start`` to ``f_end``.
    * ``"scurve"`` — raised-cosine (smooth-acceleration) sweep from
      ``f_start`` to ``f_end``.

    Both ramp shapes hold at ``f_end`` for ``t > duration_s`` (the segment
    stays valid indefinitely until the next transition replaces it).
    """

    channel: int
    tone_index: int
    shape: str  # "hold" | "linear" | "scurve"
    f_start: float
    f_end: float
    duration_s: float
    amplitude_pct: float
    phase_offset_rad: float
    static_phase_rad: float
    start_sample: int


def segment_instantaneous_phase(seg: ToneSegment, t_local, xp=np):
    """Vectorized ``2*pi * integral(f(tau), tau=0..t)`` term (rad).

    Does NOT include ``phase_offset_rad``/``static_phase_rad`` — callers add
    those separately. ``t_local`` is seconds since ``seg.start_sample``, as
    an ``xp`` array.
    """
    if seg.shape == "hold":
        return TWO_PI * seg.f_end * t_local

    duration = seg.duration_s
    t_c = xp.minimum(t_local, duration)
    tail = TWO_PI * seg.f_end * xp.maximum(t_local - duration, 0.0)

    if seg.shape == "linear":
        slope = (seg.f_end - seg.f_start) / duration
        ramp = TWO_PI * (seg.f_start * t_c + 0.5 * slope * t_c * t_c)
        return ramp + tail

    if seg.shape == "scurve":
        delta_f = seg.f_end - seg.f_start
        ramp = TWO_PI * (
            seg.f_start * t_c
            + 0.5
            * delta_f
            * (t_c - (duration / math.pi) * xp.sin(math.pi * t_c / duration))
        )
        return ramp + tail

    raise ValueError(
        f"Unknown ToneSegment.shape {seg.shape!r}; expected 'hold'/'linear'/'scurve'."
    )


def segment_total_phase(seg: ToneSegment, t_local, xp=np):
    """Full instantaneous phase (rad), including offsets."""
    return (
        seg.phase_offset_rad
        + segment_instantaneous_phase(seg, t_local, xp=xp)
        + seg.static_phase_rad
    )


def segment_instantaneous_frequency(seg: ToneSegment, t_local, xp=np):
    """Vectorized instantaneous frequency ``f(t)`` (Hz) for *seg* — the
    analytical ramp shape a tone is commanded to follow, independent of
    phase. Used for the "pure" frequency-vs-time trajectory plot
    (:func:`synthesize_round_frequency_trajectory`).

    Unlike phase, frequency is **not cumulative** — it depends only on where
    ``t_local`` falls within *this* segment's own ``duration_s``, so (unlike
    :func:`segment_instantaneous_phase`) there's no separate "tail" term:
    clamping ``t_local`` at ``duration_s`` already lands exactly on
    ``f_end`` for both ``"linear"`` and ``"scurve"``, and holds there.
    """
    if seg.shape == "hold":
        return t_local * 0.0 + seg.f_end

    duration = seg.duration_s
    t_c = xp.minimum(t_local, duration)

    if seg.shape == "linear":
        slope = (seg.f_end - seg.f_start) / duration
        return seg.f_start + slope * t_c

    if seg.shape == "scurve":
        delta_f = seg.f_end - seg.f_start
        return seg.f_start + 0.5 * delta_f * (1.0 - xp.cos(math.pi * t_c / duration))

    raise ValueError(
        f"Unknown ToneSegment.shape {seg.shape!r}; expected 'hold'/'linear'/'scurve'."
    )


def _transition_tone_segments(
    segments: Dict[Tuple[int, int], ToneSegment],
    batch: AWGBatch,
    transition_sample: int,
    sample_rate_hz: float,
    ramp_shape: str,
) -> Dict[Tuple[int, int], ToneSegment]:
    """Computes the next ``{(channel, tone_index): ToneSegment}`` state at
    ``transition_sample``, carrying each tone's phase continuously across
    the boundary: evaluates the outgoing segment's closed-form phase at the
    exact sample the incoming segment takes over, and carries that as the
    incoming segment's ``phase_offset_rad``.
    """
    shape = "hold" if batch.travel_duration_s <= 0 else ramp_shape
    new_segments: Dict[Tuple[int, int], ToneSegment] = {}
    for ramp in batch.ramps:
        key = (ramp.channel, ramp.tone_index)
        old = segments[key]
        t_old = (transition_sample - old.start_sample) / sample_rate_hz
        phase_at_transition = (
            old.phase_offset_rad
            + float(segment_instantaneous_phase(old, np.array([t_old]), xp=np)[0])
        ) % TWO_PI
        new_segments[key] = ToneSegment(
            channel=ramp.channel,
            tone_index=ramp.tone_index,
            shape=shape,
            f_start=ramp.f_start,
            f_end=ramp.f_end,
            duration_s=batch.travel_duration_s,
            amplitude_pct=ramp.amplitude_pct,
            phase_offset_rad=phase_at_transition,
            static_phase_rad=math.radians(ramp.phase_deg),
            start_sample=transition_sample,
        )
    return new_segments


def synthesize_round_waveform(
    batches: Sequence[AWGBatch],
    sample_rate_hz: float,
    *,
    ramp_shape: str = "linear",
) -> Dict[int, np.ndarray]:
    """Offline (CPU/numpy) synthesis of the per-channel waveform SCAPP would
    stream to the card for *batches* played back-to-back.

    Reuses the exact phase-continuous segment math the real-time engine
    uses, so the result matches what actually reaches the AOD — for offline
    visualization/analysis (e.g. spectrograms via
    :meth:`session_recorder.SessionRecorder.save_spectrogram`),
    not part of the real-time path itself.

    The first batch's ``f_start`` per tone is treated as the pre-existing
    hold frequency. Zero-duration batches contribute no samples but still
    update the carried phase/frequency state.

    Returns ``{channel: samples}``, float64 in roughly ``[-1, 1]`` (sum of
    unit-amplitude tones scaled by ``amplitude_pct``), one entry per channel
    seen across *batches*. Empty dict if *batches* is empty.
    """
    if not batches:
        return {}

    segments: Dict[Tuple[int, int], ToneSegment] = {
        (ramp.channel, ramp.tone_index): ToneSegment(
            channel=ramp.channel,
            tone_index=ramp.tone_index,
            shape="hold",
            f_start=ramp.f_start,
            f_end=ramp.f_start,
            duration_s=0.0,
            amplitude_pct=ramp.amplitude_pct,
            phase_offset_rad=0.0,
            static_phase_rad=math.radians(ramp.phase_deg),
            start_sample=0,
        )
        for ramp in batches[0].ramps
    }

    channels = sorted({ramp.channel for batch in batches for ramp in batch.ramps})
    chunks: Dict[int, list] = {ch: [] for ch in channels}
    next_sample = 0

    for batch in batches:
        n_samples = int(round(batch.travel_duration_s * sample_rate_hz))
        segments = _transition_tone_segments(
            segments, batch, next_sample, sample_rate_hz, ramp_shape
        )
        if n_samples > 0:
            abs_sample = next_sample + np.arange(n_samples, dtype=np.int64)
            for ch in channels:
                total = np.zeros(n_samples, dtype=np.float64)
                for (seg_ch, _tone_idx), seg in segments.items():
                    if seg_ch != ch:
                        continue
                    t_local = (abs_sample - seg.start_sample).astype(
                        np.float64
                    ) / sample_rate_hz
                    phase = segment_total_phase(seg, t_local, xp=np)
                    total = total + np.sin(phase) * (seg.amplitude_pct / 100.0)
                chunks[ch].append(total)
            next_sample += n_samples

    return {
        ch: (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64))
        for ch, parts in chunks.items()
    }


def synthesize_round_frequency_trajectory(
    batches: Sequence[AWGBatch],
    *,
    ramp_shape: str = "linear",
    points_per_batch: int = 64,
    break_tol_hz: float = 1.0,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """Analytical (phase/FFT-free) per-tone frequency trajectory for
    *batches* played back-to-back — the exact ``f(t)`` ramp shape
    (:func:`segment_instantaneous_frequency`) each tone is commanded to
    follow, i.e. the *design intent* rather than the synthesized waveform.

    Distinct from :func:`synthesize_round_waveform`, which sums sine tones
    for FFT/spectrogram analysis: instantaneous frequency isn't cumulative
    across batches (unlike phase), so this needs no ``ToneSegment``
    phase-continuity bookkeeping — each batch's ramp is fully described by
    its own ``(f_start, f_end, duration_s, shape)``.

    A tone's ``f_start`` in one batch doesn't always match its ``f_end`` in
    the previous batch it appeared in — e.g. ``RFConverter.convert_moves``
    rebuilds every non-targeted tone's ramp from its *nominal* resting
    frequency each batch, regardless of where an earlier batch in the same
    round actually left it, so a tone that moved once and isn't re-targeted
    gets a real, physically-commanded frequency jump on the next batch
    boundary. Drawing a straight connecting line across that jump would
    misrepresent it as a continuous ramp, so a ``NaN`` separator is inserted
    into that tone's trajectory instead — matplotlib breaks the line there
    rather than drawing it.

    Returns ``{(channel, tone_index): (times_s, freqs_hz)}`` spanning the
    full round (cumulative time across *batches*), one entry per tone seen
    in *batches* — including non-moving tones, which naturally evaluate flat
    since ``f_start == f_end`` collapses both ramp shapes to a constant.
    ``points_per_batch`` sets the plotting resolution per batch (a
    zero-duration/holding batch contributes a single point). ``break_tol_hz``
    is the mismatch threshold for inserting a break.
    """
    trajectories: Dict[Tuple[int, int], Tuple[list, list]] = {}
    last_f_end: Dict[Tuple[int, int], float] = {}
    t_offset = 0.0
    for batch in batches:
        duration = max(float(batch.travel_duration_s), 0.0)
        shape = "hold" if duration <= 0 else ramp_shape
        t_local = (
            np.linspace(0.0, duration, max(int(points_per_batch), 2))
            if duration > 0
            else np.zeros(1)
        )
        for ramp in batch.ramps:
            key = (ramp.channel, ramp.tone_index)
            times_list, freqs_list = trajectories.setdefault(key, ([], []))
            prev_end = last_f_end.get(key)
            if prev_end is not None and abs(ramp.f_start - prev_end) > break_tol_hz:
                times_list.append(np.array([t_offset]))
                freqs_list.append(np.array([np.nan]))

            seg = ToneSegment(
                channel=ramp.channel,
                tone_index=ramp.tone_index,
                shape=shape,
                f_start=ramp.f_start,
                f_end=ramp.f_end,
                duration_s=duration,
                amplitude_pct=ramp.amplitude_pct,
                phase_offset_rad=0.0,
                static_phase_rad=0.0,
                start_sample=0,
            )
            freqs = segment_instantaneous_frequency(seg, t_local, xp=np)
            times_list.append(t_offset + t_local)
            freqs_list.append(freqs)
            last_f_end[key] = float(ramp.f_end)
        t_offset += duration

    return {
        key: (np.concatenate(ts), np.concatenate(fs))
        for key, (ts, fs) in trajectories.items()
    }
