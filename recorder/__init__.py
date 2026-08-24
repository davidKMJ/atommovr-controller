"""Bundled ``RoundHook``: writes ``meta.json`` at session start and appends
one JSON record per round to ``rounds.jsonl``. Not part of the control loop
-- attach via ``AodController(..., hooks=[Recorder(...)])``.
"""

from recorder.recorder import Recorder, moves_to_records

__all__ = [
    "Recorder",
    "moves_to_records",
]
