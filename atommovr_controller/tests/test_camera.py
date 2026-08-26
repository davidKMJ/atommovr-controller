from __future__ import annotations

import numpy as np

from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams
from atommovr.utils.errormodels import ZeroNoise

from awg_controller.awg_control import AODSettings

from atommovr_controller.camera import GaussianCameraConfig, OfflineArrayCamera, RealArrayCamera
from atommovr_controller.controller import AtommovrController, HardwareConfig, SoftwareConfig


class TestOfflineArrayCamera:
    def test_acquire_uses_generator_shape(self):
        gen = GaussianCameraConfig(image_shape=(120, 160), min_spacing_px=12.0)
        cam = OfflineArrayCamera(
            (4, 4),
            image_generator=gen,
            seed=0,
            initial_occupancy=np.ones((4, 4), dtype=int),
        )
        img = cam.acquire()
        assert img.ndim == 2
        assert img.dtype == np.uint8
        assert cam.occupancy is not None
        assert cam.occupancy.sum() == 16

    def test_array_moves_update_camera_occupancy_via_set_occupancy(self):
        """Physics simulation (evaluate_moves) now lives on the controller's
        own AtomArray, not on the camera; the camera's occupancy is pushed
        forward from that array via set_occupancy()."""
        initial = np.zeros((5, 5), dtype=int)
        initial[0, 0] = 1
        initial[1, 1] = 1
        initial[2, 2] = 1
        initial[4, 4] = 1

        cam = OfflineArrayCamera(
            (5, 5),
            image_generator=GaussianCameraConfig(
                image_shape=(128, 128), min_spacing_px=16.0, noise_level=0.0
            ),
            physical_params=PhysicalParams(loading_prob=1.0),
            initial_occupancy=initial,
            seed=0,
        )
        before = cam.occupancy.copy()
        img0 = cam.acquire()
        assert np.array_equal(cam.occupancy, before)

        array = AtomArray((5, 5), n_species=1, error_model=ZeroNoise(seed=0))
        array.matrix[:, :, 0] = initial
        array.evaluate_moves([[Move(0, 0, 0, 1)]])
        after = (array.matrix[:, :, 0] > 0).astype(int)
        assert after[0, 0] == 0
        assert after[0, 1] == 1
        assert not np.array_equal(after, before)

        cam.set_occupancy(after)
        img1 = cam.acquire()
        assert img1.shape == img0.shape
        assert np.array_equal(cam.occupancy, after)

    def test_sync_bootstraps_on_first_call_then_absorbs_array_truth(self):
        """Round 0: camera's own occupancy is measured into the array (the
        array isn't pulled from, since it has no ground truth yet). Later
        calls: the camera absorbs the array's ground truth first."""
        cam = OfflineArrayCamera(
            (4, 4),
            image_generator=GaussianCameraConfig(
                image_shape=(128, 128), min_spacing_px=16.0, noise_level=0.0
            ),
            initial_occupancy=np.ones((4, 4), dtype=int),
            seed=0,
        )
        array = AtomArray((4, 4), n_species=1, error_model=ZeroNoise(seed=0))
        assert array.matrix.sum() == 0

        cam.sync(array)
        # Camera's own (fully-occupied) initial state was measured into array.
        assert array.matrix[:, :, 0].sum() == 16

        # Now advance ground truth directly on the array (as evaluate_moves
        # would) and sync again: the camera should absorb it before re-measuring.
        array.matrix[0, 0, 0] = 0
        cam.sync(array)
        assert cam.occupancy.sum() == 15

    def test_sync_stashes_last_frame(self):
        """sync() stashes the measured frame on last_frame so the controller
        can put it on the round context for hooks to read."""
        cam = OfflineArrayCamera(
            (4, 4),
            image_generator=GaussianCameraConfig(
                image_shape=(128, 128), min_spacing_px=16.0, noise_level=0.0
            ),
            initial_occupancy=np.ones((4, 4), dtype=int),
            seed=0,
        )
        assert cam.last_frame is None
        array = AtomArray((4, 4), n_species=1, error_model=ZeroNoise(seed=0))
        cam.sync(array)
        assert cam.last_frame is not None
        assert cam.last_frame.ndim == 2

    def test_real_array_camera_measures_into_array(self):
        """RealArrayCamera wraps a raw hardware callback and writes a
        detected reading into the caller-owned array via sync()."""
        gen = GaussianCameraConfig(
            image_shape=(128, 128), min_spacing_px=16.0, noise_level=0.0
        )
        occ = np.ones((4, 4), dtype=int)

        cam = RealArrayCamera((4, 4), camera_fn=lambda: gen(occ))
        array = AtomArray((4, 4), n_species=1)
        assert array.matrix.sum() == 0

        cam.sync(array)
        assert array.matrix[:, :, 0].sum() == 16

    def test_read_power_returns_grid_intensities(self):
        gen = GaussianCameraConfig(
            image_shape=(128, 128), min_spacing_px=16.0, noise_level=0.0
        )
        occ = np.ones((4, 4), dtype=int)
        occ[0, 0] = 0
        occ[3, 2] = 0
        cam = RealArrayCamera((4, 4), camera_fn=lambda: gen(occ))
        frame = cam.acquire()
        powers = cam.read_power(frame)
        assert powers.shape == (4, 4)
        assert powers[0, 0] == 0.0
        assert powers[3, 2] == 0.0
        occupied = powers[occ.astype(bool)]
        assert occupied.size == 14
        assert np.all(occupied > 0)

    def test_controller_multi_round_occupancy_continuity(self):
        rows, cols = 6, 5
        target = (2, 2)
        rng = np.random.default_rng(123)
        initial = (rng.random((rows, cols)) < 0.85).astype(int)

        offline = OfflineArrayCamera(
            (rows, cols),
            image_generator=GaussianCameraConfig(
                image_shape=(200, 200), min_spacing_px=20.0, noise_level=0.0
            ),
            physical_params=PhysicalParams(loading_prob=0.85, spacing=5e-6),
            initial_occupancy=initial,
            seed=1,
        )
        sw = SoftwareConfig(
            max_rounds=3,
            algorithm_name="Hungarian",
            error_model=ZeroNoise(seed=1),
        )
        hw = HardwareConfig(
            physical_params=PhysicalParams(
                loading_prob=0.85, spacing=5e-6, middle_size=list(target)
            ),
            aod_settings=AODSettings(grid_rows=rows, grid_cols=cols),
        )
        with AtommovrController(sw, hw, camera=offline) as ctrl:
            assert ctrl.camera is offline
            assert ctrl.grid_shape == (rows, cols)
            occ_snapshots = []
            _orig = offline.set_occupancy

            def _wrap(occ):
                _orig(occ)
                occ_snapshots.append(np.asarray(occ, dtype=int).copy())

            offline.set_occupancy = _wrap  # type: ignore[method-assign]
            ctrl.run()

        if occ_snapshots:
            # First post-move update should preserve atom count under ZeroNoise
            assert occ_snapshots[0].sum() == initial.sum()
            other = (np.random.default_rng(999).random((rows, cols)) < 0.85).astype(int)
            assert not np.array_equal(occ_snapshots[0], other)

    def test_explicit_offline_camera_used_directly(self):
        """An explicitly injected camera= should be used as-is, not wrapped."""
        gen = GaussianCameraConfig(
            image_shape=(128, 128), min_spacing_px=16.0, noise_level=0.0
        )
        offline = OfflineArrayCamera(
            (4, 4),
            image_generator=gen,
            initial_occupancy=np.ones((4, 4), dtype=int),
            seed=0,
        )
        sw = SoftwareConfig()
        hw = HardwareConfig(
            physical_params=PhysicalParams(middle_size=[2, 2]),
            aod_settings=AODSettings(grid_rows=4, grid_cols=4),
        )
        with AtommovrController(sw, hw, camera=offline) as ctrl:
            assert ctrl.camera is offline
