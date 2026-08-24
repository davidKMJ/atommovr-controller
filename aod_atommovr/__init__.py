"""Closed-loop AOD rearrangement controller: imaging + camera + hook bus.

The atommovr connection -- wires ``atommovr`` algorithms and ``AtomArray``
to a ``Camera`` and ``awg_controller.RFConverter`` through
:class:`AodController`, with observers (logging, recording, metrics) fanned
out by a :class:`~aod_atommovr.hooks.HookBus`.
"""

from aod_atommovr.camera import (
    Camera,
    GaussianCameraConfig,
    OfflineArrayCamera,
    RealArrayCamera,
)
from aod_atommovr.controller import (
    AodController,
    HardwareConfig,
    SoftwareConfig,
)
from aod_atommovr.hooks import Hook, HookBus, RoundContext, RoundHook, SessionContext

__all__ = [
    "AodController",
    "HardwareConfig",
    "SoftwareConfig",
    "Camera",
    "GaussianCameraConfig",
    "OfflineArrayCamera",
    "RealArrayCamera",
    "Hook",
    "HookBus",
    "RoundHook",
    "RoundContext",
    "SessionContext",
]
