"""Round-hook bus for :class:`~atommovr_controller.controller.AtommovrController`.

Everything that wants to observe a session -- logging, recording,
metrics -- is a :class:`RoundHook`, fired by :class:`HookBus` with a
read-only snapshot (:class:`RoundContext` / :class:`SessionContext`), never
the live ``AtomArray``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """Session-level metadata available at ``on_session_start``/``on_session_end``."""

    grid_shape: Tuple[int, int]
    algorithm_name: str
    max_rounds: int


@dataclass
class RoundContext:
    """Read-only snapshot of one round, passed to :class:`RoundHook` methods.

    ``before_round`` fields (``occupancy``/``target``/``frame``/``atoms``/
    ``filled``/``need``) are always populated. The ``after_round``-only
    fields default to ``None``/``0`` and are filled in before ``after_round``
    fires; ``before_round`` observers must not read them.
    """

    round_idx: int
    occupancy: np.ndarray
    target: np.ndarray
    frame: Optional[np.ndarray]
    atoms: int
    filled: int
    need: int
    # after_round only; unset (None / 0 / False-ish) on before_round.
    move_batches: Optional[list] = None
    rf_batches: Optional[list] = None
    n_moves: int = 0
    success: Optional[bool] = None
    aborted: Optional[str] = None
    total_travel_duration_s: float = 0.0


class RoundHook(Protocol):
    """Duck-typed observer. Any subset of these methods may be implemented;
    :class:`HookBus` skips whichever are missing.
    """

    def on_session_start(self, ctx: SessionContext) -> None: ...
    def before_round(self, ctx: RoundContext) -> None: ...
    def after_round(self, ctx: RoundContext) -> None: ...
    def on_session_end(self, ctx: SessionContext) -> None: ...


_PHASES = ("on_session_start", "before_round", "after_round", "on_session_end")


class Hook:
    """Wrap plain callables into a :class:`RoundHook`, one per phase.

    ``Hook(after_round=fn)`` calls ``fn(ctx)`` only on ``after_round``; pass
    any subset of the four phase kwargs. A bare callable registered directly
    in a controller's ``hooks=`` list is treated as ``Hook(after_round=fn)``
    (see :func:`HookBus`'s handling of bare callables).
    """

    def __init__(
        self,
        *,
        on_session_start: Optional[Callable[[SessionContext], None]] = None,
        before_round: Optional[Callable[[RoundContext], None]] = None,
        after_round: Optional[Callable[[RoundContext], None]] = None,
        on_session_end: Optional[Callable[[SessionContext], None]] = None,
    ) -> None:
        self._fns = {
            "on_session_start": on_session_start,
            "before_round": before_round,
            "after_round": after_round,
            "on_session_end": on_session_end,
        }

    def on_session_start(self, ctx: SessionContext) -> None:
        fn = self._fns["on_session_start"]
        if fn is not None:
            fn(ctx)

    def before_round(self, ctx: RoundContext) -> None:
        fn = self._fns["before_round"]
        if fn is not None:
            fn(ctx)

    def after_round(self, ctx: RoundContext) -> None:
        fn = self._fns["after_round"]
        if fn is not None:
            fn(ctx)

    def on_session_end(self, ctx: SessionContext) -> None:
        fn = self._fns["on_session_end"]
        if fn is not None:
            fn(ctx)


def _normalize(observer: Any) -> Any:
    """Bare callables (no hook-phase methods of their own) become
    ``Hook(after_round=observer)``; anything already exposing a hook-phase
    method (a real ``RoundHook``) passes through unchanged."""
    if callable(observer) and not any(hasattr(observer, p) for p in _PHASES):
        return Hook(after_round=observer)
    return observer


class HookBus:
    """Fan session/round events out to duck-typed :class:`RoundHook` observers.

    Missing methods are skipped. Observer exceptions are logged and
    swallowed -- a broken hook must never abort a round (unlike the camera,
    which is a required collaborator, not a hook; its failures do abort --
    see ``controller.py``).
    """

    def __init__(self, hooks: Sequence[Any] = ()) -> None:
        self._hooks: List[Any] = [_normalize(h) for h in hooks]

    def _dispatch(self, phase: str, ctx: Any) -> None:
        for hook in self._hooks:
            fn = getattr(hook, phase, None)
            if fn is None:
                continue
            try:
                fn(ctx)
            except Exception:
                log.exception("Hook %r.%s raised; continuing.", hook, phase)

    def on_session_start(self, ctx: SessionContext) -> None:
        self._dispatch("on_session_start", ctx)

    def before_round(self, ctx: RoundContext) -> None:
        self._dispatch("before_round", ctx)

    def after_round(self, ctx: RoundContext) -> None:
        self._dispatch("after_round", ctx)

    def on_session_end(self, ctx: SessionContext) -> None:
        self._dispatch("on_session_end", ctx)
