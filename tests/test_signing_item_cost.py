"""Tests for result signing and item-level cost attribution."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.models import EvalRun, MetricResult
from evalyn_sdk.evaluation.signing import (
    compute_result_hash, sign_run, verify_run,
)
from evalyn_sdk.analysis.item_cost import (
    compute_item_costs, ItemCost, ItemCostReport,
)


def _result(mid="m", iid="i1", score=1.0, passed=True, input_tokens=0,
            output_tokens=0, model=None):
    return MetricResult(
        metric_id=mid, item_id=iid, call_id="c1",
        score=score, passed=passed, details={},
        input_tokens=input_tokens, output_tokens=output_tokens, model=model,
    )


def _run(results=None, summary=None):
    now = datetime.now(timezone.utc)
    return EvalRun(
        id="run-1", dataset_name="test", created_at=now,
        metric_results=results or [], summary=summary or {},
    )


# =============================================================================
# Result signing tests
# =============================================================================

class TestComputeResultHash:
    def test_deterministic(self):
        results = [_result("a", "i1", 0.8), _result("b", "i2", 0.9)]
        h1 = compute_result_hash(results)
        h2 = compute_result_hash(results)
        assert h1 == h2

    def test_order_independent(self):
        r1 = [_result("a", "i1"), _result("b", "i2")]
        r2 = [_result("b", "i2"), _result("a", "i1")]
        assert compute_result_hash(r1) == compute_result_hash(r2)

    def test_different_scores(self):
        r1 = [_result("a", "i1", 0.8)]
        r2 = [_result("a", "i1", 0.9)]
        assert compute_result_hash(r1) != compute_result_hash(r2)

    def test_different_passed(self):
        r1 = [_result("a", "i1", passed=True)]
        r2 = [_result("a", "i1", passed=False)]
        assert compute_result_hash(r1) != compute_result_hash(r2)

    def test_empty(self):
        h = compute_result_hash([])
        assert len(h) == 32

    def test_hash_length(self):
        assert len(compute_result_hash([_result()])) == 32


class TestSignRun:
    def test_sign_stores_hash(self):
        run = _run([_result("a", "i1")])
        h = sign_run(run)
        assert run.summary["result_hash"] == h
        assert len(h) == 32

    def test_sign_empty_run(self):
        run = _run([])
        h = sign_run(run)
        assert len(h) == 32


class TestVerifyRun:
    def test_valid_run(self):
        run = _run([_result("a", "i1", 0.8)])
        sign_run(run)
        result = verify_run(run)
        assert result["valid"] is True

    def test_tampered_run(self):
        run = _run([_result("a", "i1", 0.8)])
        sign_run(run)
        # Tamper: change a score
        run.metric_results[0] = _result("a", "i1", 0.99)
        result = verify_run(run)
        assert result["valid"] is False
        assert "modified" in result["reason"]

    def test_unsigned_run(self):
        run = _run([_result()])
        result = verify_run(run)
        assert result["valid"] is None
        assert "not signed" in result["reason"]

    def test_verify_after_adding_result(self):
        run = _run([_result("a", "i1")])
        sign_run(run)
        run.metric_results.append(_result("b", "i2"))
        result = verify_run(run)
        assert result["valid"] is False


# =============================================================================
# Item cost attribution tests
# =============================================================================

class TestComputeItemCosts:
    def test_basic(self):
        results = [
            _result("m1", "i1", input_tokens=500, output_tokens=100),
            _result("m2", "i1", input_tokens=300, output_tokens=50),
            _result("m1", "i2", input_tokens=400, output_tokens=80),
        ]
        report = compute_item_costs(results)
        assert len(report.items) == 2
        # i1 should be more expensive (more tokens)
        i1 = next(ic for ic in report.items if ic.item_id == "i1")
        assert i1.input_tokens == 800
        assert i1.output_tokens == 150
        assert i1.metric_count == 2

    def test_no_tokens_free(self):
        results = [_result("m", "i1")]
        report = compute_item_costs(results)
        assert report.items[0].cost_usd == 0.0

    def test_total_cost(self):
        results = [
            _result("m", "i1", input_tokens=1000, output_tokens=500, model="gemini-2.5-flash-lite"),
            _result("m", "i2", input_tokens=2000, output_tokens=1000, model="gemini-2.5-flash-lite"),
        ]
        report = compute_item_costs(results)
        assert report.total_cost_usd > 0
        assert report.total_tokens == 4500

    def test_most_expensive(self):
        results = [
            _result("m", "i1", input_tokens=100, output_tokens=50),
            _result("m", "i2", input_tokens=10000, output_tokens=5000),
        ]
        report = compute_item_costs(results)
        assert report.most_expensive.item_id == "i2"

    def test_avg_cost(self):
        results = [
            _result("m", "i1", input_tokens=500, output_tokens=100),
            _result("m", "i2", input_tokens=500, output_tokens=100),
        ]
        report = compute_item_costs(results)
        assert report.avg_cost_per_item > 0

    def test_empty(self):
        report = compute_item_costs([])
        assert len(report.items) == 0
        assert report.most_expensive is None
        assert report.avg_cost_per_item == 0.0

    def test_top_n(self):
        results = [_result("m", f"i{i}", input_tokens=i*100, output_tokens=i*50) for i in range(10)]
        report = compute_item_costs(results)
        top = report.top_n_expensive(3)
        assert len(top) == 3
        # Highest cost first
        assert top[0].total_tokens >= top[1].total_tokens

    def test_as_dict(self):
        results = [_result("m", "i1", input_tokens=100, output_tokens=50)]
        d = compute_item_costs(results).as_dict()
        assert "total_cost_usd" in d
        assert "items" in d
        assert "most_expensive_item" in d

    def test_format_text(self):
        results = [_result("m", "i1", input_tokens=100, output_tokens=50)]
        text = compute_item_costs(results).format_text()
        assert "Item-Level Cost" in text

    def test_item_cost_as_dict(self):
        ic = ItemCost(item_id="i1", input_tokens=100, output_tokens=50,
                     total_tokens=150, cost_usd=0.001, metric_count=2)
        d = ic.as_dict()
        assert d["item_id"] == "i1"
        assert d["total_tokens"] == 150
