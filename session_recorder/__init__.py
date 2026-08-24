"""Bundled ``RoundHook``: per-round frame/occupancy dumps, rounds.jsonl,
and optional lattice GIF/SVG visualization. Not part of the control loop --
attach via ``AodController(..., hooks=[SessionRecorder(...)])``.
"""

from session_recorder.recorder import (
    GifOptions,
    SessionRecorder,
    VisualizationOptions,
    moves_to_records,
)

__all__ = [
    "SessionRecorder",
    "GifOptions",
    "VisualizationOptions",
    "moves_to_records",
]
