from __future__ import annotations

from evalyn_sdk.trace.cost_budget import (
    BudgetConfig,
    BudgetStatus,
    BudgetTracker,
    create_budget,
)


# --- BudgetConfig ---


def test_config_defaults():
    cfg = BudgetConfig(max_cost_usd=10.0)
    assert cfg.warning_threshold == 0.8
    assert cfg.hard_stop is True


def test_config_as_dict():
    cfg = BudgetConfig(max_cost_usd=5.0, warning_threshold=0.9, hard_stop=False)
    d = cfg.as_dict()
    assert d["max_cost_usd"] == 5.0
    assert d["warning_threshold"] == 0.9
    assert d["hard_stop"] is False


def test_config_from_dict():
    d = {"max_cost_usd": 7.0, "warning_threshold": 0.5, "hard_stop": False}
    cfg = BudgetConfig.from_dict(d)
    assert cfg.max_cost_usd == 7.0
    assert cfg.warning_threshold == 0.5
    assert cfg.hard_stop is False


def test_config_roundtrip():
    cfg = BudgetConfig(max_cost_usd=3.0, warning_threshold=0.6, hard_stop=True)
    restored = BudgetConfig.from_dict(cfg.as_dict())
    assert restored.max_cost_usd == cfg.max_cost_usd
    assert restored.warning_threshold == cfg.warning_threshold
    assert restored.hard_stop == cfg.hard_stop


# --- BudgetStatus ---


def test_status_as_dict():
    s = BudgetStatus(
        current_cost_usd=1.0,
        max_cost_usd=10.0,
        remaining_usd=9.0,
        utilization=0.1,
        status="ok",
        alerts=[],
    )
    d = s.as_dict()
    assert d["status"] == "ok"
    assert d["utilization"] == 0.1


def test_status_format_text_ok():
    s = BudgetStatus(
        current_cost_usd=1.0,
        max_cost_usd=10.0,
        remaining_usd=9.0,
        utilization=0.1,
        status="ok",
    )
    text = s.format_text()
    assert "ok" in text
    assert "$1.0000" in text
    assert "$10.0000" in text


def test_status_format_text_with_alerts():
    s = BudgetStatus(
        current_cost_usd=9.0,
        max_cost_usd=10.0,
        remaining_usd=1.0,
        utilization=0.9,
        status="warning",
        alerts=["Budget warning: 90% used"],
    )
    text = s.format_text()
    assert "ALERT" in text
    assert "90%" in text


# --- BudgetTracker - status levels ---


def test_ok_when_under_limit():
    t = BudgetTracker(BudgetConfig(max_cost_usd=10.0))
    t.add_cost(1.0)
    status = t.get_status()
    assert status.status == "ok"
    assert status.utilization < 0.8
    assert len(status.alerts) == 0


def test_warning_at_80_pct():
    t = BudgetTracker(BudgetConfig(max_cost_usd=10.0))
    t.add_cost(8.0)
    status = t.get_status()
    assert status.status == "warning"
    assert len(status.alerts) == 1


def test_exceeded_at_100_pct():
    t = BudgetTracker(BudgetConfig(max_cost_usd=10.0))
    t.add_cost(10.0)
    status = t.get_status()
    assert status.status == "exceeded"
    assert len(status.alerts) == 1


def test_exceeded_over_100_pct():
    t = BudgetTracker(BudgetConfig(max_cost_usd=10.0))
    t.add_cost(15.0)
    status = t.get_status()
    assert status.status == "exceeded"
    assert status.remaining_usd == 0.0


# --- should_stop ---


def test_should_stop_hard_stop_true():
    t = BudgetTracker(BudgetConfig(max_cost_usd=5.0, hard_stop=True))
    t.add_cost(6.0)
    assert t.should_stop() is True


def test_should_stop_hard_stop_false():
    t = BudgetTracker(BudgetConfig(max_cost_usd=5.0, hard_stop=False))
    t.add_cost(6.0)
    assert t.should_stop() is False


def test_should_stop_not_exceeded():
    t = BudgetTracker(BudgetConfig(max_cost_usd=10.0, hard_stop=True))
    t.add_cost(3.0)
    assert t.should_stop() is False


# --- add_cost / accumulation ---


def test_add_cost_accumulates():
    t = BudgetTracker(BudgetConfig(max_cost_usd=100.0))
    t.add_cost(1.0)
    t.add_cost(2.0)
    t.add_cost(3.0)
    status = t.get_status()
    assert abs(status.current_cost_usd - 6.0) < 1e-9


def test_per_run_costs_tracked():
    t = BudgetTracker(BudgetConfig(max_cost_usd=100.0))
    t.add_cost(1.0, run_id="run-a")
    t.add_cost(2.0, run_id="run-a")
    t.add_cost(3.0, run_id="run-b")
    assert t._run_costs["run-a"] == 3.0
    assert t._run_costs["run-b"] == 3.0


def test_add_cost_no_run_id():
    t = BudgetTracker(BudgetConfig(max_cost_usd=100.0))
    t.add_cost(5.0)
    assert t._run_costs == {}
    assert t.get_status().current_cost_usd == 5.0


# --- reset ---


def test_reset_clears_costs():
    t = BudgetTracker(BudgetConfig(max_cost_usd=10.0))
    t.add_cost(5.0, run_id="r1")
    t.reset()
    status = t.get_status()
    assert status.current_cost_usd == 0.0
    assert t._run_costs == {}


# --- check alias ---


def test_check_alias():
    t = BudgetTracker(BudgetConfig(max_cost_usd=10.0))
    t.add_cost(9.0)
    s1 = t.get_status()
    s2 = t.check()
    assert s1.status == s2.status
    assert s1.utilization == s2.utilization


# --- Tracker as_dict / from_dict ---


def test_tracker_as_dict():
    t = BudgetTracker(BudgetConfig(max_cost_usd=20.0))
    t.add_cost(3.0, run_id="r1")
    d = t.as_dict()
    assert d["total_cost"] == 3.0
    assert d["run_costs"]["r1"] == 3.0
    assert d["config"]["max_cost_usd"] == 20.0


def test_tracker_from_dict():
    d = {
        "config": {"max_cost_usd": 10.0, "warning_threshold": 0.7, "hard_stop": False},
        "total_cost": 4.0,
        "run_costs": {"x": 2.0, "y": 2.0},
    }
    t = BudgetTracker.from_dict(d)
    assert t._total_cost == 4.0
    assert t._run_costs == {"x": 2.0, "y": 2.0}
    assert t.config.warning_threshold == 0.7


def test_tracker_roundtrip():
    t = BudgetTracker(BudgetConfig(max_cost_usd=50.0, warning_threshold=0.75))
    t.add_cost(10.0, run_id="a")
    t.add_cost(5.0, run_id="b")
    restored = BudgetTracker.from_dict(t.as_dict())
    assert restored._total_cost == t._total_cost
    assert restored._run_costs == t._run_costs
    assert restored.config.max_cost_usd == t.config.max_cost_usd


# --- create_budget factory ---


def test_create_budget():
    t = create_budget(25.0)
    assert t.config.max_cost_usd == 25.0
    assert t.config.warning_threshold == 0.8


def test_create_budget_custom_threshold():
    t = create_budget(10.0, warning_threshold=0.5)
    assert t.config.warning_threshold == 0.5


# --- utilization edge cases ---


def test_utilization_zero_max():
    t = BudgetTracker(BudgetConfig(max_cost_usd=0.0))
    status = t.get_status()
    assert status.utilization == 0.0


def test_custom_warning_threshold():
    t = BudgetTracker(BudgetConfig(max_cost_usd=10.0, warning_threshold=0.5))
    t.add_cost(5.0)
    status = t.get_status()
    assert status.status == "warning"
