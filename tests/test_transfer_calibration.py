"""Tests for transfer calibration module."""

from evalyn_sdk.calibration.transfer import (
    MetricSimilarity,
    TransferReport,
    TransferResult,
    compute_rubric_similarity,
    find_similar_metrics,
    plan_transfers,
    transfer_preamble,
    validate_transfer,
)


# ---------------------------------------------------------------------------
# compute_rubric_similarity
# ---------------------------------------------------------------------------


def test_similarity_identical():
    assert compute_rubric_similarity("hello world", "hello world") == 1.0


def test_similarity_no_overlap():
    assert compute_rubric_similarity("cat dog", "fish bird") == 0.0


def test_similarity_partial_overlap():
    score = compute_rubric_similarity("the cat sat", "the dog sat")
    # intersection = {the, sat}, union = {the, cat, sat, dog} -> 2/4 = 0.5
    assert score == 0.5


def test_similarity_case_insensitive():
    assert compute_rubric_similarity("Hello World", "hello world") == 1.0


def test_similarity_empty_both():
    assert compute_rubric_similarity("", "") == 0.0


def test_similarity_empty_one():
    assert compute_rubric_similarity("hello", "") == 0.0


def test_similarity_duplicate_words():
    # "a a b" -> {a, b}, "a b c" -> {a, b, c} -> intersection {a,b}/union {a,b,c} = 2/3
    score = compute_rubric_similarity("a a b", "a b c")
    assert abs(score - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# find_similar_metrics
# ---------------------------------------------------------------------------


def test_find_similar_sorted():
    rubrics = {
        "low": "x y z",
        "high": "the cat sat on the mat",
        "mid": "the dog sat",
    }
    results = find_similar_metrics("the cat sat", rubrics, min_similarity=0.0)
    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_find_similar_respects_min_similarity():
    rubrics = {
        "low": "x y z",
        "high": "the cat sat on the mat",
    }
    results = find_similar_metrics("the cat sat", rubrics, min_similarity=0.5)
    names = [r.target_metric for r in results]
    assert "high" in names
    assert "low" not in names


def test_find_similar_empty_targets():
    results = find_similar_metrics("hello world", {}, min_similarity=0.0)
    assert results == []


def test_find_similar_method_default():
    rubrics = {"x": "hello world"}
    results = find_similar_metrics("hello world", rubrics, min_similarity=0.0)
    assert results[0].method == "word_overlap"


# ---------------------------------------------------------------------------
# transfer_preamble
# ---------------------------------------------------------------------------


def test_transfer_preamble_substitutes_name():
    prompt = "Evaluate metric: helpfulness for this response."
    result = transfer_preamble(prompt, "accuracy")
    assert "accuracy" in result
    assert "helpfulness" not in result


def test_transfer_preamble_fallback():
    prompt = "Please evaluate the response carefully."
    result = transfer_preamble(prompt, "accuracy")
    assert "accuracy" in result
    assert prompt in result


def test_transfer_preamble_case_insensitive_pattern():
    prompt = "Metric: coherence is important."
    result = transfer_preamble(prompt, "fluency")
    assert "fluency" in result


# ---------------------------------------------------------------------------
# validate_transfer
# ---------------------------------------------------------------------------


def test_validate_transfer_perfect():
    scores = [0.9, 0.1, 0.8, 0.2]
    truth = [True, False, True, False]
    assert validate_transfer("prompt", scores, truth) == 1.0


def test_validate_transfer_half():
    scores = [0.9, 0.9, 0.1, 0.1]
    truth = [True, False, True, False]
    # 0.9>=0.5->True==True ok, 0.9>=0.5->True!=False wrong,
    # 0.1<0.5->False!=True wrong, 0.1<0.5->False==False ok
    assert validate_transfer("prompt", scores, truth) == 0.5


def test_validate_transfer_empty():
    assert validate_transfer("prompt", [], []) == 0.0


def test_validate_transfer_length_mismatch():
    assert validate_transfer("prompt", [0.5], [True, False]) == 0.0


def test_validate_transfer_threshold_boundary():
    # score exactly 0.5 should be treated as True
    assert validate_transfer("prompt", [0.5], [True]) == 1.0
    assert validate_transfer("prompt", [0.5], [False]) == 0.0


# ---------------------------------------------------------------------------
# plan_transfers
# ---------------------------------------------------------------------------


def test_plan_transfers_builds_report():
    sources = {"helpfulness": "evaluate metric: helpfulness carefully"}
    targets = {"accuracy": "evaluate metric: accuracy carefully"}
    report = plan_transfers(sources, targets, min_similarity=0.3)
    assert isinstance(report, TransferReport)
    assert report.total >= 0
    assert report.successful == len(report.transfers)


def test_plan_transfers_no_matches():
    sources = {"helpfulness": "abc def"}
    targets = {"accuracy": "xyz uvw"}
    report = plan_transfers(sources, targets, min_similarity=0.9)
    assert report.total == 0
    assert report.transfers == []


# ---------------------------------------------------------------------------
# TransferReport format_text
# ---------------------------------------------------------------------------


def test_transfer_report_format_text():
    t = TransferResult(
        source_metric="src",
        target_metric="tgt",
        transferred_preamble="prompt",
        similarity=0.8,
        validation_score=0.9,
    )
    report = TransferReport(transfers=[t], successful=1, total=1)
    text = report.format_text()
    assert "1/1" in text
    assert "src" in text
    assert "tgt" in text


def test_transfer_report_format_text_no_validation():
    t = TransferResult(
        source_metric="src",
        target_metric="tgt",
        transferred_preamble="prompt",
        similarity=0.6,
    )
    report = TransferReport(transfers=[t], successful=1, total=1)
    text = report.format_text()
    assert "ok" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


def test_metric_similarity_as_dict():
    ms = MetricSimilarity("a", "b", 0.75, "word_overlap")
    d = ms.as_dict()
    assert d["source_metric"] == "a"
    assert d["similarity_score"] == 0.75


def test_transfer_result_roundtrip():
    original = TransferResult(
        source_metric="src",
        target_metric="tgt",
        transferred_preamble="prompt text",
        similarity=0.65,
        validation_score=0.88,
    )
    d = original.as_dict()
    restored = TransferResult.from_dict(d)
    assert restored.source_metric == original.source_metric
    assert restored.target_metric == original.target_metric
    assert restored.transferred_preamble == original.transferred_preamble
    assert restored.similarity == original.similarity
    assert restored.validation_score == original.validation_score


def test_transfer_result_roundtrip_no_validation():
    original = TransferResult(
        source_metric="s",
        target_metric="t",
        transferred_preamble="p",
        similarity=0.5,
    )
    d = original.as_dict()
    restored = TransferResult.from_dict(d)
    assert restored.validation_score is None


def test_transfer_report_roundtrip():
    t = TransferResult(
        source_metric="src",
        target_metric="tgt",
        transferred_preamble="prompt",
        similarity=0.8,
        validation_score=0.9,
    )
    original = TransferReport(transfers=[t], successful=1, total=1)
    d = original.as_dict()
    restored = TransferReport.from_dict(d)
    assert restored.successful == 1
    assert restored.total == 1
    assert len(restored.transfers) == 1
    assert restored.transfers[0].source_metric == "src"


def test_transfer_report_roundtrip_empty():
    original = TransferReport(transfers=[], successful=0, total=0)
    d = original.as_dict()
    restored = TransferReport.from_dict(d)
    assert restored.transfers == []
    assert restored.successful == 0
