from __future__ import annotations

import numpy as np
import pytest

from atommovr_controller.hooks import Hook, HookBus, RoundContext, SessionContext


def _ctx(round_idx=0):
    occ = np.zeros((2, 2), dtype=int)
    return RoundContext(
        round_idx=round_idx,
        occupancy=occ,
        target=occ,
        frame=None,
        atoms=0,
        filled=0,
        need=0,
    )


def _session_ctx():
    return SessionContext(grid_shape=(2, 2), algorithm_name="PCFA", max_rounds=1)


class TestBareCallable:
    def test_bare_callable_is_after_round_only(self):
        calls = []
        bus = HookBus([lambda ctx: calls.append(("after", ctx.round_idx))])

        session = _session_ctx()
        bus.on_session_start(session)
        bus.before_round(_ctx(0))
        bus.after_round(_ctx(0))
        bus.on_session_end(session)

        assert calls == [("after", 0)]


class TestHookWrapper:
    def test_hook_wraps_selected_phases_only(self):
        calls = []
        hook = Hook(
            before_round=lambda ctx: calls.append(("before", ctx.round_idx)),
            after_round=lambda ctx: calls.append(("after", ctx.round_idx)),
        )
        bus = HookBus([hook])

        bus.on_session_start(_session_ctx())
        bus.before_round(_ctx(1))
        bus.after_round(_ctx(1))
        bus.on_session_end(_session_ctx())

        assert calls == [("before", 1), ("after", 1)]

    def test_hook_missing_phase_is_noop(self):
        hook = Hook(after_round=lambda ctx: None)
        bus = HookBus([hook])
        # None of these should raise even though only after_round is set.
        bus.on_session_start(_session_ctx())
        bus.before_round(_ctx(0))
        bus.on_session_end(_session_ctx())


class TestDuckTypedObserver:
    def test_partial_hook_class_only_dispatches_implemented_methods(self):
        calls = []

        class OnlyAfterRound:
            def after_round(self, ctx):
                calls.append(ctx.round_idx)

        bus = HookBus([OnlyAfterRound()])
        bus.on_session_start(_session_ctx())
        bus.before_round(_ctx(0))
        bus.after_round(_ctx(0))
        bus.on_session_end(_session_ctx())

        assert calls == [0]

    def test_full_hook_class_dispatches_all_four_phases(self):
        calls = []

        class FullHook:
            def on_session_start(self, ctx):
                calls.append("start")

            def before_round(self, ctx):
                calls.append("before")

            def after_round(self, ctx):
                calls.append("after")

            def on_session_end(self, ctx):
                calls.append("end")

        bus = HookBus([FullHook()])
        bus.on_session_start(_session_ctx())
        bus.before_round(_ctx(0))
        bus.after_round(_ctx(0))
        bus.on_session_end(_session_ctx())

        assert calls == ["start", "before", "after", "end"]


class TestObserverFailureIsolation:
    def test_raising_hook_is_logged_and_does_not_propagate(self):
        calls = []

        def _boom(ctx):
            raise RuntimeError("observer exploded")

        def _after(ctx):
            calls.append(ctx.round_idx)

        bus = HookBus([Hook(after_round=_boom), Hook(after_round=_after)])
        # Must not raise, and the second (working) hook must still run.
        bus.after_round(_ctx(3))
        assert calls == [3]

    def test_multiple_observers_all_fire(self):
        calls = []
        bus = HookBus(
            [
                Hook(after_round=lambda ctx: calls.append("a")),
                Hook(after_round=lambda ctx: calls.append("b")),
            ]
        )
        bus.after_round(_ctx(0))
        assert calls == ["a", "b"]


class TestEmptyBus:
    def test_no_hooks_is_noop(self):
        bus = HookBus([])
        bus.on_session_start(_session_ctx())
        bus.before_round(_ctx(0))
        bus.after_round(_ctx(0))
        bus.on_session_end(_session_ctx())


class TestRoundContextDefaults:
    def test_after_round_fields_default_unset(self):
        ctx = _ctx(0)
        assert ctx.move_batches is None
        assert ctx.rf_batches is None
        assert ctx.n_moves == 0
        assert ctx.success is None
        assert ctx.aborted is None
        assert ctx.total_travel_duration_s == 0.0
