"""Tests for calibration.prompt_versioning module."""

from __future__ import annotations

from evalyn_sdk.calibration.prompt_versioning import (
    PromptHistory,
    PromptVersion,
    PromptVersionStore,
    compute_prompt_hash,
    has_prompt_changed,
)


# ---------------------------------------------------------------------------
# compute_prompt_hash
# ---------------------------------------------------------------------------


def test_compute_prompt_hash_deterministic():
    h1 = compute_prompt_hash("Rate the helpfulness")
    h2 = compute_prompt_hash("Rate the helpfulness")
    assert h1 == h2


def test_compute_prompt_hash_length():
    h = compute_prompt_hash("anything")
    assert len(h) == 16


def test_compute_prompt_hash_differs_for_different_text():
    h1 = compute_prompt_hash("prompt A")
    h2 = compute_prompt_hash("prompt B")
    assert h1 != h2


# ---------------------------------------------------------------------------
# has_prompt_changed
# ---------------------------------------------------------------------------


def test_has_prompt_changed_true():
    h = compute_prompt_hash("old prompt")
    assert has_prompt_changed(h, "new prompt") is True


def test_has_prompt_changed_false():
    text = "same prompt"
    h = compute_prompt_hash(text)
    assert has_prompt_changed(h, text) is False


# ---------------------------------------------------------------------------
# PromptVersion as_dict / from_dict
# ---------------------------------------------------------------------------


def test_prompt_version_as_dict():
    pv = PromptVersion(version=1, prompt_text="hello", prompt_hash="abc", notes="first")
    d = pv.as_dict()
    assert d["version"] == 1
    assert d["prompt_text"] == "hello"
    assert d["prompt_hash"] == "abc"
    assert d["notes"] == "first"
    assert d["alignment_score"] is None


def test_prompt_version_from_dict():
    d = {"version": 2, "prompt_text": "world", "prompt_hash": "def", "alignment_score": 0.95}
    pv = PromptVersion.from_dict(d)
    assert pv.version == 2
    assert pv.prompt_text == "world"
    assert pv.alignment_score == 0.95


def test_prompt_version_roundtrip():
    pv = PromptVersion(version=3, prompt_text="x", prompt_hash="h", notes="n", alignment_score=0.8)
    pv2 = PromptVersion.from_dict(pv.as_dict())
    assert pv2.version == pv.version
    assert pv2.prompt_text == pv.prompt_text
    assert pv2.alignment_score == pv.alignment_score


# ---------------------------------------------------------------------------
# PromptHistory
# ---------------------------------------------------------------------------


def test_prompt_history_empty():
    ph = PromptHistory(metric_id="m1")
    assert ph.current is None
    assert ph.count == 0


def test_prompt_history_current():
    v1 = PromptVersion(version=1, prompt_text="a")
    v2 = PromptVersion(version=2, prompt_text="b")
    ph = PromptHistory(metric_id="m1", versions=[v1, v2])
    assert ph.current is v2
    assert ph.count == 2


def test_prompt_history_as_dict_from_dict():
    v1 = PromptVersion(version=1, prompt_text="a", prompt_hash="ha")
    ph = PromptHistory(metric_id="m1", versions=[v1])
    d = ph.as_dict()
    ph2 = PromptHistory.from_dict(d)
    assert ph2.metric_id == "m1"
    assert ph2.count == 1
    assert ph2.versions[0].prompt_text == "a"


def test_prompt_history_format_text():
    v1 = PromptVersion(version=1, prompt_text="a", prompt_hash="ha", alignment_score=0.9, notes="init")
    ph = PromptHistory(metric_id="helpfulness", versions=[v1])
    text = ph.format_text()
    assert "helpfulness" in text
    assert "1 versions" in text
    assert "v1" in text
    assert "score=0.9" in text
    assert "init" in text


def test_prompt_history_format_text_no_score_no_notes():
    v1 = PromptVersion(version=1, prompt_text="a", prompt_hash="ha")
    ph = PromptHistory(metric_id="m1", versions=[v1])
    text = ph.format_text()
    assert "score" not in text


# ---------------------------------------------------------------------------
# PromptVersionStore - add_version
# ---------------------------------------------------------------------------


def test_store_add_version_auto_increments():
    store = PromptVersionStore()
    v1 = store.add_version("m1", "prompt v1")
    v2 = store.add_version("m1", "prompt v2")
    assert v1.version == 1
    assert v2.version == 2


def test_store_add_version_sets_hash():
    store = PromptVersionStore()
    v = store.add_version("m1", "some prompt")
    assert v.prompt_hash == compute_prompt_hash("some prompt")


def test_store_add_version_sets_created_at():
    store = PromptVersionStore()
    v = store.add_version("m1", "text")
    assert v.created_at != ""


def test_store_add_version_with_notes_and_score():
    store = PromptVersionStore()
    v = store.add_version("m1", "text", notes="tweaked", alignment_score=0.85)
    assert v.notes == "tweaked"
    assert v.alignment_score == 0.85


# ---------------------------------------------------------------------------
# PromptVersionStore - queries
# ---------------------------------------------------------------------------


def test_store_get_history_creates_if_missing():
    store = PromptVersionStore()
    h = store.get_history("new_metric")
    assert h.metric_id == "new_metric"
    assert h.count == 0


def test_store_get_current_returns_latest():
    store = PromptVersionStore()
    store.add_version("m1", "first")
    store.add_version("m1", "second")
    current = store.get_current("m1")
    assert current is not None
    assert current.prompt_text == "second"


def test_store_get_current_none_for_empty():
    store = PromptVersionStore()
    assert store.get_current("absent") is None


def test_store_get_version_specific():
    store = PromptVersionStore()
    store.add_version("m1", "first")
    store.add_version("m1", "second")
    v1 = store.get_version("m1", 1)
    assert v1 is not None
    assert v1.prompt_text == "first"


def test_store_get_version_none_for_missing():
    store = PromptVersionStore()
    assert store.get_version("m1", 99) is None


def test_store_diff_versions():
    store = PromptVersionStore()
    store.add_version("m1", "line one\nline two")
    store.add_version("m1", "line one\nline changed")
    diff = store.diff_versions("m1", 1, 2)
    assert "---" in diff
    assert "+++" in diff
    assert "-line two" in diff
    assert "+line changed" in diff


def test_store_diff_versions_returns_empty_for_missing():
    store = PromptVersionStore()
    assert store.diff_versions("m1", 1, 2) == ""


def test_store_list_metrics():
    store = PromptVersionStore()
    store.add_version("b_metric", "p")
    store.add_version("a_metric", "p")
    assert store.list_metrics() == ["a_metric", "b_metric"]


def test_store_list_metrics_empty():
    store = PromptVersionStore()
    assert store.list_metrics() == []


# ---------------------------------------------------------------------------
# PromptVersionStore - serialisation
# ---------------------------------------------------------------------------


def test_store_to_json_from_json():
    store = PromptVersionStore()
    store.add_version("m1", "prompt text", notes="initial", alignment_score=0.9)
    store.add_version("m2", "other prompt")

    json_str = store.to_json()
    store2 = PromptVersionStore.from_json(json_str)

    assert store2.list_metrics() == ["m1", "m2"]
    assert store2.get_current("m1").prompt_text == "prompt text"
    assert store2.get_current("m1").alignment_score == 0.9
    assert store2.get_current("m2").prompt_text == "other prompt"


def test_store_as_dict_from_dict():
    store = PromptVersionStore()
    store.add_version("m1", "hello")
    d = store.as_dict()
    store2 = PromptVersionStore.from_dict(d)
    assert store2.get_current("m1").prompt_text == "hello"


def test_store_from_dict_empty():
    store = PromptVersionStore.from_dict({})
    assert store.list_metrics() == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_store_operations():
    store = PromptVersionStore()
    assert store.list_metrics() == []
    assert store.get_current("x") is None
    assert store.get_version("x", 1) is None
    assert store.diff_versions("x", 1, 2) == ""


def test_single_version_diff():
    store = PromptVersionStore()
    store.add_version("m1", "only version")
    # diff between v1 and non-existent v2 returns empty
    assert store.diff_versions("m1", 1, 2) == ""
