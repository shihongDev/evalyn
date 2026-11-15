"""
Tests for LogprobsConfidence, including low-confidence spans.

Run with: pytest tests/test_logprobs_confidence.py -v
"""

from __future__ import annotations

import math

import pytest
from evalyn_sdk.judges.confidence import LogprobsConfidence
from evalyn_sdk.judges.confidence import logprobs as logprobs_module

# A judge verdict where the model hesitated on the score digit and on a name
# in the rationale. Everything else is near-certain.
TOKENS = [
    '{"',
    "score",
    '":',
    " 4",
    ",",
    ' "',
    "reason",
    '":',
    ' "',
    "The",
    " answer",
    " cites",
    " Marco",
    " Pant",
    "ani",
    '."}',
]
LOGPROBS = [
    -0.01,
    -0.01,
    -0.01,
    -1.2,
    -0.01,
    -0.01,
    -0.01,
    -0.01,
    -0.01,
    -0.05,
    -0.1,
    -0.2,
    -0.9,
    -0.02,
    -0.01,
    -0.01,
]


class TestLogprobsScore:
    def test_score_is_geometric_mean(self):
        result = LogprobsConfidence().estimate(logprobs=[-0.1, -0.3, -0.2])
        assert result.score == pytest.approx(math.exp(-0.2))
        assert "low_confidence_spans" not in result.details

    def test_empty_logprobs_unchanged(self):
        result = LogprobsConfidence().estimate(logprobs=[], tokens=[])
        assert result.score == 0.5
        assert result.details == {"reason": "no_logprobs"}

    def test_tokens_do_not_change_score(self):
        plain = LogprobsConfidence().estimate(logprobs=LOGPROBS)
        with_tokens = LogprobsConfidence().estimate(logprobs=LOGPROBS, tokens=TOKENS)
        assert with_tokens.score == plain.score

    def test_rejects_threshold_outside_unit_interval(self):
        with pytest.raises(ValueError, match="span_threshold"):
            LogprobsConfidence(span_threshold=50)


class TestLowConfidenceSpans:
    @pytest.fixture(autouse=True)
    def _require_ltv(self):
        pytest.importorskip("llm_token_visualizer")

    def test_flags_the_hesitant_words(self):
        details = LogprobsConfidence().estimate(logprobs=LOGPROBS, tokens=TOKENS).details
        assert details["flagged"] is True
        spans = details["low_confidence_spans"]
        assert [s["text"] for s in spans] == [" 4", " Marco"]
        assert spans[0]["min_prob"] == pytest.approx(math.exp(-1.2))
        assert (spans[1]["token_start"], spans[1]["token_end"]) == (12, 13)
        # A scalar score of ~0.89 alone would not say where the doubt is.
        assert LogprobsConfidence().estimate(logprobs=LOGPROBS).score > 0.85

    def test_confident_answer_has_no_spans(self):
        details = (
            LogprobsConfidence().estimate(logprobs=[-0.01] * len(TOKENS), tokens=TOKENS).details
        )
        assert details["flagged"] is False
        assert details["low_confidence_spans"] == []

    def test_span_threshold(self):
        details = (
            LogprobsConfidence(span_threshold=0.35)
            .estimate(logprobs=LOGPROBS, tokens=TOKENS)
            .details
        )
        assert [s["text"] for s in details["low_confidence_spans"]] == [" 4"]

    def test_dict_top_logprobs_name_alternatives(self):
        top = [None] * len(TOKENS)
        top[12] = {" Marco": -0.9, " Jan": -0.97}
        details = (
            LogprobsConfidence()
            .estimate(logprobs=LOGPROBS, tokens=TOKENS, top_logprobs=top)
            .details
        )
        marco = details["low_confidence_spans"][1]
        assert [alt[0] for alt in marco["alternatives"]] == [" Marco", " Jan"]

    def test_float_top_logprobs_are_ignored_for_spans(self):
        top = [[-0.1, -2.3]] * len(TOKENS)
        details = (
            LogprobsConfidence()
            .estimate(logprobs=LOGPROBS, tokens=TOKENS, top_logprobs=top)
            .details
        )
        assert all(s["alternatives"] == [] for s in details["low_confidence_spans"])

    def test_none_logprob_is_kept_in_text(self):
        details = (
            LogprobsConfidence()
            .estimate(logprobs=[-0.1, None, -2.0], tokens=["a", " b", " c"])
            .details
        )
        assert [s["text"] for s in details["low_confidence_spans"]] == [" c"]

    def test_single_token(self):
        details = LogprobsConfidence().estimate(logprobs=[-2.0], tokens=[" Paris"]).details
        assert [s["text"] for s in details["low_confidence_spans"]] == [" Paris"]

    def test_large_input(self):
        n = 10_000
        lps = [-0.01] * n
        lps[n // 2] = -3.0
        toks = [f" w{i}" for i in range(n)]
        details = LogprobsConfidence().estimate(logprobs=lps, tokens=toks).details
        assert [s["token_start"] for s in details["low_confidence_spans"]] == [n // 2]

    def test_length_mismatch_is_reported(self):
        details = LogprobsConfidence().estimate(logprobs=LOGPROBS, tokens=TOKENS[:3]).details
        assert "low_confidence_spans" not in details
        assert "3 tokens for 16 logprobs" in details["spans_error"]


def test_missing_package_is_reported(monkeypatch):
    monkeypatch.setattr(logprobs_module, "_ltv", None)
    result = LogprobsConfidence().estimate(logprobs=LOGPROBS, tokens=TOKENS)
    assert "evalyn-sdk[spans]" in result.details["spans_error"]
    assert result.score == LogprobsConfidence().estimate(logprobs=LOGPROBS).score
