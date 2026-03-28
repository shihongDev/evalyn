"""Tests for provider_fallback module."""

from __future__ import annotations

from evalyn_sdk.evaluation.provider_fallback import (
    FallbackAttempt,
    FallbackChainConfig,
    FallbackResult,
    ProviderConfig,
    build_fallback_report,
    create_fallback_chain,
    execute_with_fallback,
    get_ordered_providers,
    suggest_chain_optimization,
)


# -- ProviderConfig --


def test_provider_config_defaults():
    p = ProviderConfig(name="openai")
    assert p.name == "openai"
    assert p.model == ""
    assert p.priority == 0
    assert p.max_retries == 1
    assert p.enabled is True


def test_provider_config_as_dict():
    p = ProviderConfig(name="openai", model="gpt-4", priority=1, max_retries=3, enabled=False)
    d = p.as_dict()
    assert d["name"] == "openai"
    assert d["model"] == "gpt-4"
    assert d["priority"] == 1
    assert d["max_retries"] == 3
    assert d["enabled"] is False


def test_provider_config_from_dict():
    d = {"name": "anthropic", "model": "claude-3", "priority": 2, "max_retries": 2, "enabled": True}
    p = ProviderConfig.from_dict(d)
    assert p.name == "anthropic"
    assert p.model == "claude-3"
    assert p.priority == 2
    assert p.max_retries == 2


def test_provider_config_from_dict_defaults():
    p = ProviderConfig.from_dict({"name": "test"})
    assert p.model == ""
    assert p.priority == 0
    assert p.max_retries == 1
    assert p.enabled is True


def test_provider_config_roundtrip():
    p = ProviderConfig(name="openai", model="gpt-4", priority=5, max_retries=3, enabled=False)
    restored = ProviderConfig.from_dict(p.as_dict())
    assert restored.name == p.name
    assert restored.model == p.model
    assert restored.priority == p.priority
    assert restored.max_retries == p.max_retries
    assert restored.enabled == p.enabled


# -- FallbackAttempt --


def test_fallback_attempt_defaults():
    a = FallbackAttempt(provider="openai", success=True)
    assert a.provider == "openai"
    assert a.success is True
    assert a.error == ""
    assert a.duration_ms == 0.0


def test_fallback_attempt_as_dict():
    a = FallbackAttempt(provider="openai", success=False, error="timeout", duration_ms=150.5)
    d = a.as_dict()
    assert d["provider"] == "openai"
    assert d["success"] is False
    assert d["error"] == "timeout"
    assert d["duration_ms"] == 150.5


# -- FallbackResult --


def test_fallback_result_defaults():
    r = FallbackResult()
    assert r.final_provider == ""
    assert r.attempts == []
    assert r.success is False
    assert r.total_attempts == 0


def test_fallback_result_as_dict():
    a = FallbackAttempt(provider="openai", success=True, duration_ms=100.0)
    r = FallbackResult(final_provider="openai", attempts=[a], success=True, total_attempts=1)
    d = r.as_dict()
    assert d["final_provider"] == "openai"
    assert d["success"] is True
    assert d["total_attempts"] == 1
    assert len(d["attempts"]) == 1
    assert d["attempts"][0]["provider"] == "openai"


def test_fallback_result_from_dict():
    d = {
        "final_provider": "anthropic",
        "attempts": [
            {"provider": "openai", "success": False, "error": "err", "duration_ms": 50.0},
            {"provider": "anthropic", "success": True, "error": "", "duration_ms": 100.0},
        ],
        "success": True,
        "total_attempts": 2,
    }
    r = FallbackResult.from_dict(d)
    assert r.final_provider == "anthropic"
    assert r.success is True
    assert r.total_attempts == 2
    assert len(r.attempts) == 2
    assert r.attempts[0].provider == "openai"
    assert r.attempts[1].success is True


def test_fallback_result_from_dict_defaults():
    r = FallbackResult.from_dict({})
    assert r.final_provider == ""
    assert r.success is False
    assert r.attempts == []


def test_fallback_result_roundtrip():
    a1 = FallbackAttempt(provider="openai", success=False, error="fail", duration_ms=50.0)
    a2 = FallbackAttempt(provider="anthropic", success=True, duration_ms=200.0)
    r = FallbackResult(final_provider="anthropic", attempts=[a1, a2], success=True, total_attempts=2)
    restored = FallbackResult.from_dict(r.as_dict())
    assert restored.final_provider == r.final_provider
    assert restored.success == r.success
    assert restored.total_attempts == r.total_attempts
    assert len(restored.attempts) == 2
    assert restored.attempts[0].error == "fail"
    assert restored.attempts[1].provider == "anthropic"


def test_fallback_result_format_text_success():
    a = FallbackAttempt(provider="openai", success=True, duration_ms=100.0)
    r = FallbackResult(final_provider="openai", attempts=[a], success=True, total_attempts=1)
    text = r.format_text()
    assert "Success: True" in text
    assert "openai" in text
    assert "Total attempts: 1" in text


def test_fallback_result_format_text_failure():
    a = FallbackAttempt(provider="openai", success=False, error="timeout", duration_ms=50.0)
    r = FallbackResult(final_provider="openai", attempts=[a], success=False, total_attempts=1)
    text = r.format_text()
    assert "Success: False" in text
    assert "FAIL" in text
    assert "timeout" in text


# -- FallbackChainConfig --


def test_chain_config_defaults():
    cfg = FallbackChainConfig()
    assert cfg.providers == []
    assert cfg.stop_on_first_success is True


def test_chain_config_as_dict():
    p = ProviderConfig(name="openai")
    cfg = FallbackChainConfig(providers=[p], stop_on_first_success=False)
    d = cfg.as_dict()
    assert len(d["providers"]) == 1
    assert d["providers"][0]["name"] == "openai"
    assert d["stop_on_first_success"] is False


def test_chain_config_from_dict():
    d = {
        "providers": [{"name": "openai", "priority": 1}, {"name": "anthropic", "priority": 0}],
        "stop_on_first_success": False,
    }
    cfg = FallbackChainConfig.from_dict(d)
    assert len(cfg.providers) == 2
    assert cfg.providers[0].name == "openai"
    assert cfg.providers[1].name == "anthropic"
    assert cfg.stop_on_first_success is False


def test_chain_config_from_dict_defaults():
    cfg = FallbackChainConfig.from_dict({})
    assert cfg.providers == []
    assert cfg.stop_on_first_success is True


def test_chain_config_roundtrip():
    p1 = ProviderConfig(name="openai", model="gpt-4", priority=1)
    p2 = ProviderConfig(name="anthropic", model="claude-3", priority=0)
    cfg = FallbackChainConfig(providers=[p1, p2], stop_on_first_success=False)
    restored = FallbackChainConfig.from_dict(cfg.as_dict())
    assert len(restored.providers) == 2
    assert restored.providers[0].name == "openai"
    assert restored.providers[1].name == "anthropic"
    assert restored.stop_on_first_success is False


# -- create_fallback_chain --


def test_create_fallback_chain_from_dicts():
    dicts = [
        {"name": "openai", "model": "gpt-4", "priority": 1},
        {"name": "anthropic", "model": "claude-3", "priority": 0},
    ]
    cfg = create_fallback_chain(dicts)
    assert len(cfg.providers) == 2
    assert cfg.providers[0].name == "openai"
    assert cfg.providers[1].name == "anthropic"


def test_create_fallback_chain_empty():
    cfg = create_fallback_chain([])
    assert cfg.providers == []


# -- get_ordered_providers --


def test_get_ordered_providers_sorted():
    p1 = ProviderConfig(name="openai", priority=2)
    p2 = ProviderConfig(name="anthropic", priority=0)
    p3 = ProviderConfig(name="google", priority=1)
    cfg = FallbackChainConfig(providers=[p1, p2, p3])
    ordered = get_ordered_providers(cfg)
    assert [p.name for p in ordered] == ["anthropic", "google", "openai"]


def test_get_ordered_providers_filters_disabled():
    p1 = ProviderConfig(name="openai", priority=0, enabled=True)
    p2 = ProviderConfig(name="anthropic", priority=1, enabled=False)
    p3 = ProviderConfig(name="google", priority=2, enabled=True)
    cfg = FallbackChainConfig(providers=[p1, p2, p3])
    ordered = get_ordered_providers(cfg)
    assert len(ordered) == 2
    assert all(p.name != "anthropic" for p in ordered)


def test_get_ordered_providers_all_disabled():
    p1 = ProviderConfig(name="openai", enabled=False)
    cfg = FallbackChainConfig(providers=[p1])
    ordered = get_ordered_providers(cfg)
    assert ordered == []


# -- execute_with_fallback --


def test_execute_first_succeeds():
    p1 = ProviderConfig(name="openai", priority=0)
    cfg = FallbackChainConfig(providers=[p1])

    def execute_fn(provider: str, model: str) -> str:
        return "ok"

    result = execute_with_fallback(cfg, execute_fn)
    assert result.success is True
    assert result.final_provider == "openai"
    assert result.total_attempts == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].success is True


def test_execute_falls_back_on_failure():
    p1 = ProviderConfig(name="openai", priority=0)
    p2 = ProviderConfig(name="anthropic", priority=1)
    cfg = FallbackChainConfig(providers=[p1, p2])

    call_count = {"n": 0}

    def execute_fn(provider: str, model: str) -> str:
        call_count["n"] += 1
        if provider == "openai":
            raise RuntimeError("openai down")
        return "ok"

    result = execute_with_fallback(cfg, execute_fn)
    assert result.success is True
    assert result.final_provider == "anthropic"
    assert result.total_attempts == 2
    assert result.attempts[0].success is False
    assert result.attempts[0].error == "openai down"
    assert result.attempts[1].success is True


def test_execute_all_fail():
    p1 = ProviderConfig(name="openai", priority=0)
    p2 = ProviderConfig(name="anthropic", priority=1)
    cfg = FallbackChainConfig(providers=[p1, p2])

    def execute_fn(provider: str, model: str) -> str:
        raise RuntimeError(f"{provider} down")

    result = execute_with_fallback(cfg, execute_fn)
    assert result.success is False
    assert result.total_attempts == 2
    assert all(not a.success for a in result.attempts)


def test_execute_retries():
    p1 = ProviderConfig(name="openai", priority=0, max_retries=3)
    cfg = FallbackChainConfig(providers=[p1])

    call_count = {"n": 0}

    def execute_fn(provider: str, model: str) -> str:
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RuntimeError("fail")
        return "ok"

    result = execute_with_fallback(cfg, execute_fn)
    assert result.success is True
    assert result.total_attempts == 3
    assert result.attempts[0].success is False
    assert result.attempts[1].success is False
    assert result.attempts[2].success is True


def test_execute_no_providers():
    cfg = FallbackChainConfig(providers=[])

    def execute_fn(provider: str, model: str) -> str:
        return "ok"

    result = execute_with_fallback(cfg, execute_fn)
    assert result.success is False
    assert result.total_attempts == 0
    assert result.attempts == []


def test_execute_single_provider():
    p1 = ProviderConfig(name="openai", priority=0)
    cfg = FallbackChainConfig(providers=[p1])

    def execute_fn(provider: str, model: str) -> str:
        return "result"

    result = execute_with_fallback(cfg, execute_fn)
    assert result.success is True
    assert result.final_provider == "openai"
    assert result.total_attempts == 1


def test_execute_respects_priority_order():
    p1 = ProviderConfig(name="low_priority", priority=10)
    p2 = ProviderConfig(name="high_priority", priority=0)
    cfg = FallbackChainConfig(providers=[p1, p2])

    called: list = []

    def execute_fn(provider: str, model: str) -> str:
        called.append(provider)
        return "ok"

    execute_with_fallback(cfg, execute_fn)
    assert called[0] == "high_priority"


def test_execute_skips_disabled():
    p1 = ProviderConfig(name="disabled_provider", priority=0, enabled=False)
    p2 = ProviderConfig(name="enabled_provider", priority=1, enabled=True)
    cfg = FallbackChainConfig(providers=[p1, p2])

    called: list = []

    def execute_fn(provider: str, model: str) -> str:
        called.append(provider)
        return "ok"

    result = execute_with_fallback(cfg, execute_fn)
    assert result.success is True
    assert result.final_provider == "enabled_provider"
    assert "disabled_provider" not in called


def test_execute_duration_tracked():
    p1 = ProviderConfig(name="openai", priority=0)
    cfg = FallbackChainConfig(providers=[p1])

    def execute_fn(provider: str, model: str) -> str:
        return "ok"

    result = execute_with_fallback(cfg, execute_fn)
    assert result.attempts[0].duration_ms >= 0.0


# -- build_fallback_report --


def test_build_fallback_report_basic():
    a1 = FallbackAttempt(provider="openai", success=True, duration_ms=100.0)
    r1 = FallbackResult(final_provider="openai", attempts=[a1], success=True, total_attempts=1)

    a2 = FallbackAttempt(provider="openai", success=False, error="err", duration_ms=50.0)
    a3 = FallbackAttempt(provider="anthropic", success=True, duration_ms=200.0)
    r2 = FallbackResult(final_provider="anthropic", attempts=[a2, a3], success=True, total_attempts=2)

    report = build_fallback_report([r1, r2])
    assert report["total_executions"] == 2
    assert report["provider_attempts"]["openai"] == 2
    assert report["provider_attempts"]["anthropic"] == 1
    assert report["provider_success_rates"]["openai"] == 0.5
    assert report["provider_success_rates"]["anthropic"] == 1.0
    assert report["avg_fallback_depth"] == 1.5


def test_build_fallback_report_empty():
    report = build_fallback_report([])
    assert report["total_executions"] == 0
    assert report["provider_attempts"] == {}
    assert report["avg_fallback_depth"] == 0.0


# -- suggest_chain_optimization --


def test_suggest_chain_optimization_empty():
    assert suggest_chain_optimization([]) == []


def test_suggest_chain_optimization_flags_always_failing():
    a1 = FallbackAttempt(provider="bad_provider", success=False, error="err")
    a2 = FallbackAttempt(provider="good_provider", success=True)
    r1 = FallbackResult(
        final_provider="good_provider",
        attempts=[a1, a2],
        success=True,
        total_attempts=2,
    )
    r2 = FallbackResult(
        final_provider="good_provider",
        attempts=[
            FallbackAttempt(provider="bad_provider", success=False, error="err"),
            FallbackAttempt(provider="good_provider", success=True),
        ],
        success=True,
        total_attempts=2,
    )
    suggestions = suggest_chain_optimization([r1, r2])
    assert any("bad_provider" in s and "0%" in s for s in suggestions)


def test_suggest_chain_optimization_move_to_front():
    # good_provider always succeeds but is never the first attempt
    results = []
    for _ in range(10):
        a1 = FallbackAttempt(provider="first_provider", success=False, error="err")
        a2 = FallbackAttempt(provider="good_provider", success=True)
        results.append(FallbackResult(
            final_provider="good_provider",
            attempts=[a1, a2],
            success=True,
            total_attempts=2,
        ))
    suggestions = suggest_chain_optimization(results)
    assert any("good_provider" in s and "front" in s for s in suggestions)


# -- edge: single provider chain --


def test_single_provider_success():
    cfg = create_fallback_chain([{"name": "only_provider"}])
    result = execute_with_fallback(cfg, lambda p, m: "ok")
    assert result.success is True
    assert result.final_provider == "only_provider"


def test_single_provider_failure():
    cfg = create_fallback_chain([{"name": "only_provider"}])

    def fail(p: str, m: str) -> str:
        raise RuntimeError("down")

    result = execute_with_fallback(cfg, fail)
    assert result.success is False
    assert result.total_attempts == 1
