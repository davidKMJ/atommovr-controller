"""Closed-loop AOD rearrangement controller: imaging + camera + hook bus.

The atommovr connection -- wires ``atommovr`` algorithms and ``AtomArray``
to a ``Camera`` and ``awg_controller.RFConverter`` through
:class:`AtommovrController`, with observers (logging, recording, metrics) fanned
out by a :class:`~atommovr_controller.hooks.HookBus`.
"""

from atommovr_controller.camera import (
    Camera,
    GaussianCameraConfig,
    OfflineArrayCamera,
    RealArrayCamera,
)
from atommovr_controller.controller import (
    AtommovrController,
    HardwareConfig,
    SoftwareConfig,
    configure_logging,
)
from atommovr_controller.hooks import Hook, HookBus, RoundContext, RoundHook, SessionContext

__all__ = [
    "AtommovrController",
    "HardwareConfig",
    "SoftwareConfig",
    "configure_logging",
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
