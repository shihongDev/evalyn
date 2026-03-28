from __future__ import annotations

import os

from evalyn_sdk.trace.conditional import (
    TracingCondition,
    TracingDecision,
    environment_condition,
    evaluate_condition,
    evaluate_conditions,
    predicate_condition,
    sample_rate_condition,
)


# -- TracingCondition dataclass --


class TestTracingConditionRoundtrip:
    def test_as_dict_fields(self):
        c = TracingCondition(name="test", condition_type="sample_rate", config={"rate": 0.5})
        d = c.as_dict()
        assert d["name"] == "test"
        assert d["condition_type"] == "sample_rate"
        assert d["config"] == {"rate": 0.5}

    def test_from_dict_roundtrip(self):
        original = TracingCondition(
            name="env", condition_type="environment", config={"allowed_envs": ["prod"]}
        )
        restored = TracingCondition.from_dict(original.as_dict())
        assert restored.name == original.name
        assert restored.condition_type == original.condition_type
        assert restored.config == original.config

    def test_from_dict_defaults(self):
        c = TracingCondition.from_dict({"name": "x", "condition_type": "sample_rate"})
        assert c.config == {}


# -- TracingDecision dataclass --


class TestTracingDecision:
    def test_fields(self):
        d = TracingDecision(should_trace=True, reason="sampled", condition_name="c1")
        assert d.should_trace is True
        assert d.reason == "sampled"
        assert d.condition_name == "c1"

    def test_as_dict(self):
        d = TracingDecision(should_trace=False, reason="rate_excluded").as_dict()
        assert d == {
            "should_trace": False,
            "reason": "rate_excluded",
            "condition_name": "",
        }

    def test_default_condition_name(self):
        d = TracingDecision(should_trace=True, reason="ok")
        assert d.condition_name == ""


# -- sample_rate_condition --


class TestSampleRateCondition:
    def test_rate_1_always_traces(self):
        cond = sample_rate_condition(1.0, seed=42)
        for _ in range(20):
            result = evaluate_condition(cond)
            assert result.should_trace is True
            assert result.reason == "sampled"

    def test_rate_0_never_traces(self):
        cond = sample_rate_condition(0.0, seed=42)
        for _ in range(20):
            result = evaluate_condition(cond)
            assert result.should_trace is False
            assert result.reason == "rate_excluded"

    def test_creates_correct_type(self):
        cond = sample_rate_condition(0.5)
        assert cond.condition_type == "sample_rate"
        assert cond.config["rate"] == 0.5

    def test_seed_stored_in_config(self):
        cond = sample_rate_condition(0.5, seed=99)
        assert cond.config["seed"] == 99

    def test_no_seed_not_in_config(self):
        cond = sample_rate_condition(0.5)
        assert "seed" not in cond.config


# -- environment_condition --


class TestEnvironmentCondition:
    def test_matches_current_env(self, monkeypatch):
        monkeypatch.setenv("EVALYN_ENV", "production")
        cond = environment_condition(["production", "staging"])
        result = evaluate_condition(cond)
        assert result.should_trace is True
        assert "production" in result.reason

    def test_rejects_unmatched_env(self, monkeypatch):
        monkeypatch.setenv("EVALYN_ENV", "testing")
        cond = environment_condition(["production"])
        result = evaluate_condition(cond)
        assert result.should_trace is False
        assert "testing" in result.reason

    def test_defaults_to_development(self, monkeypatch):
        monkeypatch.delenv("EVALYN_ENV", raising=False)
        cond = environment_condition(["development"])
        result = evaluate_condition(cond)
        assert result.should_trace is True

    def test_default_env_rejected(self, monkeypatch):
        monkeypatch.delenv("EVALYN_ENV", raising=False)
        cond = environment_condition(["production"])
        result = evaluate_condition(cond)
        assert result.should_trace is False

    def test_creates_correct_type(self):
        cond = environment_condition(["prod"])
        assert cond.condition_type == "environment"
        assert cond.config["allowed_envs"] == ["prod"]


# -- predicate_condition --


class TestPredicateCondition:
    def test_predicate_true(self):
        cond, pred = predicate_condition("always_true", lambda: True)
        result = evaluate_condition(cond, context={"predicate": pred})
        assert result.should_trace is True
        assert result.reason == "predicate passed"

    def test_predicate_false(self):
        cond, pred = predicate_condition("always_false", lambda: False)
        result = evaluate_condition(cond, context={"predicate": pred})
        assert result.should_trace is False
        assert result.reason == "predicate failed"

    def test_predicate_with_args(self):
        cond, pred = predicate_condition("gt_check", lambda x, y: x > y)
        result = evaluate_condition(
            cond,
            context={"predicate": pred, "predicate_args": (10, 5)},
        )
        assert result.should_trace is True

    def test_no_predicate_in_context(self):
        cond, _pred = predicate_condition("orphan", lambda: True)
        result = evaluate_condition(cond, context={})
        assert result.should_trace is False
        assert "no predicate" in result.reason

    def test_creates_correct_type(self):
        cond, pred = predicate_condition("mycheck", lambda: True)
        assert cond.condition_type == "predicate"
        assert cond.name == "mycheck"
        assert callable(pred)


# -- evaluate_conditions (multi) --


class TestEvaluateConditionsAll:
    def test_all_pass(self):
        c1 = sample_rate_condition(1.0, seed=0)
        c2 = sample_rate_condition(1.0, seed=1)
        result = evaluate_conditions([c1, c2], mode="all")
        assert result.should_trace is True
        assert "all conditions passed" in result.reason

    def test_one_fails(self):
        c1 = sample_rate_condition(1.0, seed=0)
        c2 = sample_rate_condition(0.0, seed=1)
        result = evaluate_conditions([c1, c2], mode="all")
        assert result.should_trace is False
        assert "failed" in result.reason

    def test_empty_conditions(self):
        result = evaluate_conditions([], mode="all")
        assert result.should_trace is True


class TestEvaluateConditionsAny:
    def test_one_passes(self):
        c1 = sample_rate_condition(0.0, seed=0)
        c2 = sample_rate_condition(1.0, seed=1)
        result = evaluate_conditions([c1, c2], mode="any")
        assert result.should_trace is True
        assert "passed" in result.reason

    def test_none_pass(self):
        c1 = sample_rate_condition(0.0, seed=0)
        c2 = sample_rate_condition(0.0, seed=1)
        result = evaluate_conditions([c1, c2], mode="any")
        assert result.should_trace is False

    def test_all_pass_any_mode(self):
        c1 = sample_rate_condition(1.0, seed=0)
        c2 = sample_rate_condition(1.0, seed=1)
        result = evaluate_conditions([c1, c2], mode="any")
        assert result.should_trace is True


class TestEvaluateConditionsMixed:
    def test_mixed_env_and_rate(self, monkeypatch):
        monkeypatch.setenv("EVALYN_ENV", "production")
        c_env = environment_condition(["production"])
        c_rate = sample_rate_condition(1.0, seed=0)
        result = evaluate_conditions([c_env, c_rate], mode="all")
        assert result.should_trace is True

    def test_mixed_env_rejects(self, monkeypatch):
        monkeypatch.setenv("EVALYN_ENV", "testing")
        c_env = environment_condition(["production"])
        c_rate = sample_rate_condition(1.0, seed=0)
        result = evaluate_conditions([c_env, c_rate], mode="all")
        assert result.should_trace is False

    def test_unknown_mode(self):
        c1 = sample_rate_condition(1.0, seed=0)
        result = evaluate_conditions([c1], mode="bogus")
        assert result.should_trace is False
        assert "unknown mode" in result.reason
