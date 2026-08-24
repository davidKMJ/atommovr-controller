import pytest

from awg_controller.awg_engine import AWGEngineConfig


class TestAWGEngineConfig:
    def test_defaults(self):
        cfg = AWGEngineConfig()
        assert cfg.mode == "stream"
        assert cfg.notify_samples == 262144
        assert cfg.dma_buffer_samples == 16 * 1024 * 1024
        assert cfg.dma_buffer_samples % cfg.notify_samples == 0
        assert cfg.fill_start_threshold_promille == 800
        assert cfg.sample_rate_hz == 500e6
        assert cfg.ramp_shape == "linear"

    def test_memory_mode_requires_power_of_two_tail(self):
        """The park segment is looped by the card, so it must contain a whole
        number of cycles of every tone
        """
        AWGEngineConfig(mode="memory", hold_tail_samples=1 << 20)
        with pytest.raises(ValueError):
            AWGEngineConfig(mode="memory", hold_tail_samples=1000)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            AWGEngineConfig(mode="precompute")

    def test_dma_buffer_must_be_multiple_of_notify_samples(self):
        """The SCAPP DMA buffer handling only works correctly when
        dma_buffer_samples is an exact multiple of notify_samples
        """
        cfg = AWGEngineConfig()  # defaults must satisfy this themselves
        assert cfg.dma_buffer_samples % cfg.notify_samples == 0

        with pytest.raises(ValueError):
            AWGEngineConfig(notify_samples=1000, dma_buffer_samples=32_000_001)

        # A valid, non-default combination must not raise.
        AWGEngineConfig(notify_samples=1024, dma_buffer_samples=1024 * 100)

    def test_notify_samples_must_be_positive(self):
        with pytest.raises(ValueError):
            AWGEngineConfig(notify_samples=0)


class TestSimulationGuard:
    def test_awg_engine_module_imports_without_native_library(self):
        import awg_controller.awg_engine as sp

        assert sp.AWGEngine is not None
        assert sp.AWGEngineConfig is not None


class TestAWGEngineLifecycle:
    def test_methods_require_open(self):
        from awg_controller.awg_engine import AWGEngine, CardConfig

        engine = AWGEngine(CardConfig())
        with pytest.raises(RuntimeError, match=r"open\(\) has not been called"):
            engine.play()
        with pytest.raises(RuntimeError, match=r"open\(\) has not been called"):
            engine.load_round([])
        assert engine.last_error is None

    def test_amplitude_ceiling_checked_before_native(self):
        from awg_controller.awg_engine import AWGEngine, CardConfig, MAX_SAFE_OUTPUT_V

        engine = AWGEngine(CardConfig(max_amplitude_v=MAX_SAFE_OUTPUT_V + 0.1))
        with pytest.raises(ValueError, match="hard safety ceiling"):
            engine.open()
        assert engine.last_error is None
