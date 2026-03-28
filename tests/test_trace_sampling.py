from __future__ import annotations

from datetime import datetime, timedelta, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.trace.sampling import (
    SamplingConfig,
    SamplingDecision,
    create_sampler,
    should_sample,
)


def _now():
    return datetime.now(timezone.utc)


def _span(sid="s1", name="s", span_type="custom", duration_ms=100, status="ok", **attrs):
    t = _now()
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=None,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        status=status,
        attributes=attrs,
    )


# -- SamplingConfig as_dict / from_dict --


class TestSamplingConfigRoundtrip:
    def test_as_dict_contains_all_fields(self):
        cfg = SamplingConfig(rate=0.5, seed=42)
        d = cfg.as_dict()
        assert d["rate"] == 0.5
        assert d["always_capture_errors"] is True
        assert d["always_capture_slow"] is True
        assert d["slow_threshold_ms"] == 5000.0
        assert d["seed"] == 42

    def test_from_dict_roundtrip(self):
        original = SamplingConfig(
            rate=0.3,
            always_capture_errors=False,
            always_capture_slow=False,
            slow_threshold_ms=2000.0,
            seed=99,
        )
        restored = SamplingConfig.from_dict(original.as_dict())
        assert restored.rate == original.rate
        assert restored.always_capture_errors == original.always_capture_errors
        assert restored.always_capture_slow == original.always_capture_slow
        assert restored.slow_threshold_ms == original.slow_threshold_ms
        assert restored.seed == original.seed

    def test_from_dict_defaults(self):
        cfg = SamplingConfig.from_dict({"rate": 0.8})
        assert cfg.always_capture_errors is True
        assert cfg.always_capture_slow is True
        assert cfg.slow_threshold_ms == 5000.0
        assert cfg.seed is None


# -- SamplingDecision --


class TestSamplingDecision:
    def test_as_dict(self):
        d = SamplingDecision(should_capture=True, reason="sampled").as_dict()
        assert d == {"should_capture": True, "reason": "sampled"}

    def test_as_dict_excluded(self):
        d = SamplingDecision(should_capture=False, reason="rate_excluded").as_dict()
        assert d == {"should_capture": False, "reason": "rate_excluded"}


# -- should_sample --


class TestShouldSample:
    def test_rate_1_always_captures(self):
        cfg = SamplingConfig(rate=1.0, seed=0)
        for _ in range(20):
            result = should_sample(cfg)
            assert result.should_capture is True
            assert result.reason == "sampled"

    def test_rate_0_never_captures_without_span(self):
        cfg = SamplingConfig(rate=0.0, seed=0)
        for _ in range(20):
            result = should_sample(cfg)
            assert result.should_capture is False
            assert result.reason == "rate_excluded"

    def test_rate_0_captures_error_span(self):
        cfg = SamplingConfig(rate=0.0, always_capture_errors=True, seed=0)
        s = _span(status="error")
        result = should_sample(cfg, span=s)
        assert result.should_capture is True
        assert result.reason == "error_priority"

    def test_rate_0_skips_error_when_disabled(self):
        cfg = SamplingConfig(rate=0.0, always_capture_errors=False, seed=0)
        s = _span(status="error")
        result = should_sample(cfg, span=s)
        assert result.should_capture is False

    def test_rate_0_captures_slow_span(self):
        cfg = SamplingConfig(
            rate=0.0, always_capture_slow=True, slow_threshold_ms=100.0, seed=0
        )
        s = _span(duration_ms=200)
        result = should_sample(cfg, span=s)
        assert result.should_capture is True
        assert result.reason == "slow_priority"

    def test_rate_0_skips_slow_when_disabled(self):
        cfg = SamplingConfig(rate=0.0, always_capture_slow=False, seed=0)
        s = _span(duration_ms=99999)
        result = should_sample(cfg, span=s)
        assert result.should_capture is False

    def test_slow_threshold_boundary_exact(self):
        cfg = SamplingConfig(
            rate=0.0, always_capture_slow=True, slow_threshold_ms=100.0, seed=0
        )
        s = _span(duration_ms=100)
        result = should_sample(cfg, span=s)
        assert result.should_capture is True
        assert result.reason == "slow_priority"

    def test_slow_threshold_boundary_below(self):
        cfg = SamplingConfig(
            rate=0.0, always_capture_slow=True, slow_threshold_ms=100.0, seed=0
        )
        s = _span(duration_ms=99)
        result = should_sample(cfg, span=s)
        assert result.should_capture is False

    def test_error_priority_over_rate(self):
        """Error priority fires even at rate=0."""
        cfg = SamplingConfig(rate=0.0, always_capture_errors=True, seed=0)
        s = _span(status="error")
        result = should_sample(cfg, span=s)
        assert result.should_capture is True
        assert result.reason == "error_priority"

    def test_deterministic_via_create_sampler(self):
        # Deterministic sampling requires create_sampler (stateful RNG)
        # should_sample is stateless and uses global RNG
        cfg = SamplingConfig(rate=0.5, seed=12345)
        sampler_a = create_sampler(cfg)
        sampler_b = create_sampler(cfg)
        results_a = [sampler_a(None).should_capture for _ in range(10)]
        results_b = [sampler_b(None).should_capture for _ in range(10)]
        assert results_a == results_b

    def test_no_span_uses_rate(self):
        cfg = SamplingConfig(rate=1.0, seed=0)
        result = should_sample(cfg, span=None)
        assert result.should_capture is True
        assert result.reason == "sampled"


# -- create_sampler --


class TestCreateSampler:
    def test_sampler_callable_returns_decision(self):
        cfg = SamplingConfig(rate=1.0, seed=0)
        sampler = create_sampler(cfg)
        result = sampler()
        assert isinstance(result, SamplingDecision)
        assert result.should_capture is True

    def test_sampler_rate_0(self):
        cfg = SamplingConfig(rate=0.0, seed=0)
        sampler = create_sampler(cfg)
        for _ in range(10):
            assert sampler().should_capture is False

    def test_sampler_preserves_rng_state(self):
        """Two samplers with same seed produce same sequence."""
        cfg_a = SamplingConfig(rate=0.5, seed=42)
        cfg_b = SamplingConfig(rate=0.5, seed=42)
        sampler_a = create_sampler(cfg_a)
        sampler_b = create_sampler(cfg_b)
        seq_a = [sampler_a().should_capture for _ in range(20)]
        seq_b = [sampler_b().should_capture for _ in range(20)]
        assert seq_a == seq_b

    def test_sampler_error_priority(self):
        cfg = SamplingConfig(rate=0.0, always_capture_errors=True, seed=0)
        sampler = create_sampler(cfg)
        s = _span(status="error")
        result = sampler(s)
        assert result.should_capture is True
        assert result.reason == "error_priority"

    def test_sampler_slow_priority(self):
        cfg = SamplingConfig(
            rate=0.0, always_capture_slow=True, slow_threshold_ms=50.0, seed=0
        )
        sampler = create_sampler(cfg)
        s = _span(duration_ms=100)
        result = sampler(s)
        assert result.should_capture is True
        assert result.reason == "slow_priority"

    def test_sampler_roughly_half_at_rate_05(self):
        """At rate=0.5 with many samples, roughly half should be captured."""
        cfg = SamplingConfig(rate=0.5, seed=7)
        sampler = create_sampler(cfg)
        captured = sum(1 for _ in range(1000) if sampler().should_capture)
        # Allow generous tolerance
        assert 350 < captured < 650, f"Expected ~500, got {captured}"

    def test_sampler_different_seeds_differ(self):
        cfg_a = SamplingConfig(rate=0.5, seed=1)
        cfg_b = SamplingConfig(rate=0.5, seed=2)
        sampler_a = create_sampler(cfg_a)
        sampler_b = create_sampler(cfg_b)
        seq_a = [sampler_a().should_capture for _ in range(50)]
        seq_b = [sampler_b().should_capture for _ in range(50)]
        # Extremely unlikely to be identical
        assert seq_a != seq_b

    def test_sampler_ok_span_uses_rate(self):
        cfg = SamplingConfig(rate=1.0, seed=0)
        sampler = create_sampler(cfg)
        s = _span(status="ok", duration_ms=10)
        result = sampler(s)
        assert result.should_capture is True
        assert result.reason == "sampled"

    def test_sampler_no_end_time_not_slow(self):
        """Span without end_time has None duration - not slow."""
        cfg = SamplingConfig(
            rate=0.0, always_capture_slow=True, slow_threshold_ms=50.0, seed=0
        )
        sampler = create_sampler(cfg)
        t = _now()
        s = Span(
            id="x", name="x", span_type="custom",
            parent_id=None, start_time=t, end_time=None,
            status="ok", attributes={},
        )
        result = sampler(s)
        assert result.should_capture is False
        assert result.reason == "rate_excluded"
