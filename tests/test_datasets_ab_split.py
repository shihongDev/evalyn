"""Tests for dataset A/B split generator."""

import pytest
from evalyn_sdk.datasets_ab_split import (
    ABPair,
    ABSplitConfig,
    ABSplitResult,
    compute_balance_score,
    create_matched_pairs,
    render_split_summary,
    split_ab,
    validate_ab_split,
)


def _items(n: int, category: str = "") -> list:
    """Create n test items with optional category."""
    result = []
    for i in range(n):
        item = {"input": f"text input number {i}", "id": str(i)}
        if category:
            item["category"] = category
        result.append(item)
    return result


def _mixed_items(counts: dict) -> list:
    """Create items with specified category counts. counts = {'cat_a': 10, 'cat_b': 5}."""
    result = []
    idx = 0
    for cat, count in counts.items():
        for _ in range(count):
            result.append({"input": f"item {idx}", "id": str(idx), "category": cat})
            idx += 1
    return result


# ---------------------------------------------------------------------------
# ABPair
# ---------------------------------------------------------------------------


class TestABPair:
    def test_as_dict(self):
        pair = ABPair(pair_id="p0", item_a={"x": 1}, item_b={"x": 2}, similarity=0.9)
        d = pair.as_dict()
        assert d["pair_id"] == "p0"
        assert d["similarity"] == 0.9
        assert d["item_a"] == {"x": 1}

    def test_defaults(self):
        pair = ABPair(pair_id="p1")
        assert pair.item_a == {}
        assert pair.item_b == {}
        assert pair.similarity == 0.0


# ---------------------------------------------------------------------------
# ABSplitConfig
# ---------------------------------------------------------------------------


class TestABSplitConfig:
    def test_as_dict(self):
        cfg = ABSplitConfig(split_ratio=0.6, stratify_by="cat", seed=42)
        d = cfg.as_dict()
        assert d["split_ratio"] == 0.6
        assert d["stratify_by"] == "cat"
        assert d["seed"] == 42

    def test_from_dict_roundtrip(self):
        cfg = ABSplitConfig(split_ratio=0.7, stratify_by="type", match_by_length=False, seed=99)
        d = cfg.as_dict()
        restored = ABSplitConfig.from_dict(d)
        assert restored.split_ratio == 0.7
        assert restored.stratify_by == "type"
        assert restored.match_by_length is False
        assert restored.seed == 99

    def test_from_dict_defaults(self):
        cfg = ABSplitConfig.from_dict({})
        assert cfg.split_ratio == 0.5
        assert cfg.stratify_by == ""
        assert cfg.match_by_length is True
        assert cfg.seed is None


# ---------------------------------------------------------------------------
# ABSplitResult
# ---------------------------------------------------------------------------


class TestABSplitResult:
    def test_as_dict_from_dict_roundtrip(self):
        result = ABSplitResult(
            group_a=[{"id": "1"}],
            group_b=[{"id": "2"}],
            pairs=[ABPair("p0", {"id": "1"}, {"id": "2"}, 0.8)],
            balance_score=0.95,
        )
        d = result.as_dict()
        restored = ABSplitResult.from_dict(d)
        assert len(restored.group_a) == 1
        assert len(restored.group_b) == 1
        assert len(restored.pairs) == 1
        assert restored.pairs[0].pair_id == "p0"
        assert restored.balance_score == 0.95

    def test_format_text(self):
        result = ABSplitResult(
            group_a=[{"id": "1"}, {"id": "2"}],
            group_b=[{"id": "3"}],
            pairs=[],
            balance_score=0.8,
        )
        text = result.format_text()
        assert "A/B Split Result" in text
        assert "Group A: 2" in text
        assert "Group B: 1" in text
        assert "0.8" in text

    def test_format_text_empty(self):
        result = ABSplitResult()
        text = result.format_text()
        assert "Group A: 0" in text


# ---------------------------------------------------------------------------
# split_ab
# ---------------------------------------------------------------------------


class TestSplitAB:
    def test_even_split(self):
        items = _items(100)
        config = ABSplitConfig(split_ratio=0.5, seed=42)
        result = split_ab(items, config)
        assert len(result.group_a) == 50
        assert len(result.group_b) == 50
        assert result.balance_score > 0

    def test_70_30_split(self):
        items = _items(100)
        config = ABSplitConfig(split_ratio=0.7, seed=42)
        result = split_ab(items, config)
        assert len(result.group_a) == 70
        assert len(result.group_b) == 30

    def test_with_stratification(self):
        items = _mixed_items({"alpha": 40, "beta": 60})
        config = ABSplitConfig(split_ratio=0.5, stratify_by="category", seed=42)
        result = split_ab(items, config)

        # Check category proportions are maintained
        cats_a = [item["category"] for item in result.group_a]
        cats_b = [item["category"] for item in result.group_b]
        alpha_a = cats_a.count("alpha")
        beta_a = cats_a.count("beta")
        alpha_b = cats_b.count("alpha")
        beta_b = cats_b.count("beta")
        # Both groups should have items from both categories
        assert alpha_a > 0
        assert beta_a > 0
        assert alpha_b > 0
        assert beta_b > 0

    def test_deterministic_with_seed(self):
        items = _items(50)
        config = ABSplitConfig(seed=123)
        r1 = split_ab(items, config)
        r2 = split_ab(items, config)
        ids_a1 = [item["id"] for item in r1.group_a]
        ids_a2 = [item["id"] for item in r2.group_a]
        assert ids_a1 == ids_a2

    def test_different_seeds_differ(self):
        items = _items(50)
        r1 = split_ab(items, ABSplitConfig(seed=1))
        r2 = split_ab(items, ABSplitConfig(seed=999))
        ids_a1 = [item["id"] for item in r1.group_a]
        ids_a2 = [item["id"] for item in r2.group_a]
        assert ids_a1 != ids_a2

    def test_empty_items(self):
        result = split_ab([], ABSplitConfig())
        assert result.group_a == []
        assert result.group_b == []
        assert result.balance_score == 1.0

    def test_small_dataset_two_items(self):
        items = _items(2)
        config = ABSplitConfig(split_ratio=0.5, seed=42)
        result = split_ab(items, config)
        assert len(result.group_a) + len(result.group_b) == 2

    def test_odd_count(self):
        items = _items(7)
        config = ABSplitConfig(split_ratio=0.5, seed=42)
        result = split_ab(items, config)
        total = len(result.group_a) + len(result.group_b)
        assert total == 7

    def test_no_overlap(self):
        items = _items(20)
        config = ABSplitConfig(seed=42)
        result = split_ab(items, config)
        ids_a = {item["id"] for item in result.group_a}
        ids_b = {item["id"] for item in result.group_b}
        assert ids_a & ids_b == set()
        assert ids_a | ids_b == {str(i) for i in range(20)}


# ---------------------------------------------------------------------------
# compute_balance_score
# ---------------------------------------------------------------------------


class TestComputeBalanceScore:
    def test_perfectly_balanced(self):
        a = [{"input": "hello world"}] * 10
        b = [{"input": "hello there"}] * 10
        score = compute_balance_score(a, b)
        assert score > 0.8

    def test_imbalanced_sizes(self):
        a = [{"input": "x"}] * 100
        b = [{"input": "x"}] * 1
        score = compute_balance_score(a, b)
        assert score < 0.7  # heavily imbalanced size ratio pulls score down

    def test_empty_both(self):
        assert compute_balance_score([], []) == 1.0

    def test_one_empty(self):
        assert compute_balance_score([{"input": "x"}], []) == 0.0
        assert compute_balance_score([], [{"input": "x"}]) == 0.0

    def test_score_range(self):
        a = _items(20)
        b = _items(20)
        score = compute_balance_score(a, b)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# create_matched_pairs
# ---------------------------------------------------------------------------


class TestCreateMatchedPairs:
    def test_basic_pairing(self):
        items = [
            {"input": "short"},
            {"input": "also short"},
            {"input": "a much longer input text"},
            {"input": "another long input here"},
        ]
        pairs = create_matched_pairs(items)
        assert len(pairs) == 2
        for p in pairs:
            assert p.pair_id.startswith("pair_")
            assert p.similarity >= 0.0

    def test_empty_items(self):
        pairs = create_matched_pairs([])
        assert pairs == []

    def test_single_item(self):
        pairs = create_matched_pairs([{"input": "alone"}])
        assert pairs == []

    def test_odd_count(self):
        items = _items(5)
        pairs = create_matched_pairs(items)
        assert len(pairs) == 2  # 5 items -> 2 pairs, 1 leftover

    def test_similarity_high_for_similar_lengths(self):
        items = [
            {"input": "aaaa"},
            {"input": "bbbb"},
        ]
        pairs = create_matched_pairs(items)
        assert len(pairs) == 1
        assert pairs[0].similarity == 1.0


# ---------------------------------------------------------------------------
# validate_ab_split
# ---------------------------------------------------------------------------


class TestValidateABSplit:
    def test_good_split(self):
        result = ABSplitResult(
            group_a=_items(50),
            group_b=_items(50),
            balance_score=0.95,
        )
        is_valid, issues = validate_ab_split(result)
        assert is_valid is True
        assert issues == []

    def test_bad_split_low_balance(self):
        result = ABSplitResult(
            group_a=_items(50),
            group_b=_items(50),
            balance_score=0.3,
        )
        is_valid, issues = validate_ab_split(result)
        assert is_valid is False
        assert any("balance" in i.lower() for i in issues)

    def test_empty_group(self):
        result = ABSplitResult(group_a=_items(10), group_b=[], balance_score=0.0)
        is_valid, issues = validate_ab_split(result)
        assert is_valid is False
        assert any("empty" in i.lower() for i in issues)

    def test_custom_min_balance(self):
        result = ABSplitResult(
            group_a=_items(10),
            group_b=_items(10),
            balance_score=0.6,
        )
        # 0.6 passes with min_balance=0.5 but fails with 0.7
        is_valid_low, _ = validate_ab_split(result, min_balance=0.5)
        is_valid_high, _ = validate_ab_split(result, min_balance=0.7)
        assert is_valid_low is True
        assert is_valid_high is False

    def test_very_uneven_groups(self):
        result = ABSplitResult(
            group_a=_items(100),
            group_b=_items(1),
            balance_score=0.8,
        )
        is_valid, issues = validate_ab_split(result)
        assert any("uneven" in i.lower() for i in issues)


# ---------------------------------------------------------------------------
# render_split_summary
# ---------------------------------------------------------------------------


class TestRenderSplitSummary:
    def test_output_contains_key_info(self):
        result = ABSplitResult(
            group_a=_items(30),
            group_b=_items(20),
            pairs=[ABPair("p0")],
            balance_score=0.85,
        )
        text = render_split_summary(result)
        assert "A/B Split Summary" in text
        assert "Group A: 30" in text
        assert "Group B: 20" in text
        assert "50 items" in text
        assert "0.85" in text

    def test_visual_bar(self):
        result = ABSplitResult(
            group_a=_items(50),
            group_b=_items(50),
            balance_score=0.95,
        )
        text = render_split_summary(result)
        assert "[" in text and "]" in text
        assert "A" in text
        assert "B" in text

    def test_empty_result(self):
        result = ABSplitResult()
        text = render_split_summary(result)
        assert "0 items" in text

    def test_shows_pass_status(self):
        result = ABSplitResult(
            group_a=_items(10),
            group_b=_items(10),
            balance_score=0.95,
        )
        text = render_split_summary(result)
        assert "PASS" in text

    def test_shows_warn_status(self):
        result = ABSplitResult(
            group_a=_items(10),
            group_b=[],
            balance_score=0.0,
        )
        text = render_split_summary(result)
        assert "WARN" in text
