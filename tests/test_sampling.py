"""Tests for sampling module.

All tests use synthetic DatasetItems - no storage or ML deps required.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
from uuid import uuid4

import pytest

from evalyn_sdk.models import DatasetItem
from evalyn_sdk.sampling import (
    SAMPLING_MODES,
    apply_sampling,
    deduplicate,
    input_to_text,
    sample_clustered,
    sample_diverse,
    sample_random,
    sample_stratified,
    _stratification_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_item(
    query: str = "hello",
    output: str = "world",
    function: str = "agent",
    duration_ms: float = 500,
    extra_input: dict | None = None,
) -> DatasetItem:
    """Create a synthetic DatasetItem."""
    inp = {"query": query}
    if extra_input:
        inp.update(extra_input)
    return DatasetItem(
        id=str(uuid4()),
        input=inp,
        output=output,
        metadata={
            "function": function,
            "duration_ms": duration_ms,
        },
    )


def make_items(n: int) -> list[DatasetItem]:
    """Create n distinct items with unique queries."""
    return [make_item(query=f"question_{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# input_to_text
# ---------------------------------------------------------------------------


class TestInputToText:
    def test_deterministic(self):
        item = make_item(query="test")
        assert input_to_text(item) == input_to_text(item)

    def test_sorted_keys(self):
        item = DatasetItem(
            id="1",
            input={"z": 1, "a": 2},
        )
        text = input_to_text(item)
        assert text.index('"a"') < text.index('"z"')

    def test_different_inputs_different_text(self):
        a = make_item(query="alpha")
        b = make_item(query="beta")
        assert input_to_text(a) != input_to_text(b)


# ---------------------------------------------------------------------------
# sample_random
# ---------------------------------------------------------------------------


class TestSampleRandom:
    def test_returns_limit_items(self):
        items = make_items(20)
        result = sample_random(items, 5)
        assert len(result) == 5

    def test_returns_all_if_under_limit(self):
        items = make_items(3)
        result = sample_random(items, 10)
        assert len(result) == 3

    def test_deterministic_with_seed(self):
        items = make_items(50)
        r1 = sample_random(items, 10, seed=42)
        r2 = sample_random(items, 10, seed=42)
        assert [it.id for it in r1] == [it.id for it in r2]

    def test_different_seed_different_result(self):
        items = make_items(50)
        r1 = sample_random(items, 10, seed=1)
        r2 = sample_random(items, 10, seed=2)
        assert [it.id for it in r1] != [it.id for it in r2]

    def test_no_duplicates(self):
        items = make_items(30)
        result = sample_random(items, 15)
        ids = [it.id for it in result]
        assert len(ids) == len(set(ids))

    def test_empty_input(self):
        assert sample_random([], 10) == []


# ---------------------------------------------------------------------------
# _stratification_key
# ---------------------------------------------------------------------------


class TestStratificationKey:
    def test_groups_by_function(self):
        a = make_item(function="func_a")
        b = make_item(function="func_b")
        assert _stratification_key(a) != _stratification_key(b)

    def test_groups_by_duration_bucket(self):
        fast = make_item(duration_ms=50)
        slow = make_item(duration_ms=3000)
        assert _stratification_key(fast) != _stratification_key(slow)

    def test_same_group_same_key(self):
        a = make_item(function="f", duration_ms=200)
        b = make_item(function="f", duration_ms=300)
        assert _stratification_key(a) == _stratification_key(b)

    def test_duration_buckets(self):
        assert "fast" in _stratification_key(make_item(duration_ms=10))
        assert "medium" in _stratification_key(make_item(duration_ms=500))
        assert "slow" in _stratification_key(make_item(duration_ms=2000))
        assert "very_slow" in _stratification_key(make_item(duration_ms=10000))


# ---------------------------------------------------------------------------
# sample_stratified
# ---------------------------------------------------------------------------


class TestSampleStratified:
    def test_returns_limit_items(self):
        items = make_items(50)
        result = sample_stratified(items, 10)
        assert len(result) == 10

    def test_returns_all_if_under_limit(self):
        items = make_items(3)
        result = sample_stratified(items, 10)
        assert len(result) == 3

    def test_preserves_group_diversity(self):
        items = []
        for fn in ("func_a", "func_b", "func_c"):
            for _ in range(10):
                items.append(make_item(function=fn))
        result = sample_stratified(items, 9)
        functions = {it.metadata["function"] for it in result}
        # All 3 function groups should be represented
        assert len(functions) == 3

    def test_deterministic(self):
        items = make_items(30)
        r1 = sample_stratified(items, 10, seed=42)
        r2 = sample_stratified(items, 10, seed=42)
        assert [it.id for it in r1] == [it.id for it in r2]

    def test_empty_input(self):
        assert sample_stratified([], 10) == []


# ---------------------------------------------------------------------------
# deduplicate (with mocked embeddings)
# ---------------------------------------------------------------------------


class TestDeduplicate:
    def test_empty_input(self):
        assert deduplicate([]) == []

    def test_single_item(self):
        items = [make_item()]
        assert len(deduplicate(items)) == 1

    def test_no_deps_returns_original(self):
        """Without sentence-transformers, returns original list."""
        items = make_items(5)
        with patch(
            "evalyn_sdk.sampling._get_embeddings",
            side_effect=ImportError("no module"),
        ):
            result = deduplicate(items)
            assert len(result) == 5

    def test_removes_duplicates_with_mock_embeddings(self):
        """With mock embeddings showing high similarity, items get removed."""
        import numpy as np

        items = make_items(4)
        # Items 0 and 1 are near-duplicates (sim > 0.95)
        embeddings = np.array([
            [1.0, 0.0],
            [0.99, 0.01],  # very similar to item 0
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        with patch("evalyn_sdk.sampling._get_embeddings", return_value=embeddings):
            result = deduplicate(items, threshold=0.95)
            # Item 1 should be removed (too similar to item 0)
            assert len(result) == 3

    def test_keeps_all_unique_with_mock_embeddings(self):
        """Orthogonal embeddings: all items kept."""
        import numpy as np

        items = make_items(3)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        with patch("evalyn_sdk.sampling._get_embeddings", return_value=embeddings):
            result = deduplicate(items, threshold=0.95)
            assert len(result) == 3


# ---------------------------------------------------------------------------
# sample_diverse (with mocked embeddings)
# ---------------------------------------------------------------------------


class TestSampleDiverse:
    def test_returns_all_if_under_limit(self):
        items = make_items(3)
        result = sample_diverse(items, 10)
        assert len(result) == 3

    def test_no_deps_falls_back_to_random(self):
        items = make_items(20)
        with patch(
            "evalyn_sdk.sampling._get_embeddings",
            side_effect=ImportError("no module"),
        ):
            result = sample_diverse(items, 5)
            assert len(result) == 5

    def test_selects_spread_with_mock_embeddings(self):
        """Farthest-point should pick spread-out points from different regions."""
        import numpy as np

        items = make_items(6)
        # 3 clusters: (0,1) near origin, (2,3) near (5,0), (4,5) near (0,5)
        embeddings = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 0.0],
            [5.1, 0.1],
            [0.0, 5.0],
            [0.1, 5.1],
        ])
        with patch("evalyn_sdk.sampling._get_embeddings", return_value=embeddings):
            result = sample_diverse(items, 3)
            assert len(result) == 3
            # Result should have no duplicates
            result_ids = [it.id for it in result]
            assert len(result_ids) == len(set(result_ids))

    def test_empty_input(self):
        assert sample_diverse([], 10) == []


# ---------------------------------------------------------------------------
# sample_clustered (with mocked embeddings + KMeans)
# ---------------------------------------------------------------------------


class TestSampleClustered:
    def test_returns_all_if_under_limit(self):
        items = make_items(3)
        result = sample_clustered(items, 10)
        assert len(result) == 3

    def test_no_deps_falls_back_to_random(self):
        items = make_items(20)
        with patch(
            "evalyn_sdk.sampling._get_embeddings",
            side_effect=ImportError("no module"),
        ):
            result = sample_clustered(items, 5)
            assert len(result) == 5

    def test_returns_limit_items_with_mock(self):
        import numpy as np

        items = make_items(20)
        embeddings = np.random.RandomState(42).randn(20, 8)
        with patch("evalyn_sdk.sampling._get_embeddings", return_value=embeddings):
            result = sample_clustered(items, 10, n_clusters=3)
            assert len(result) == 10

    def test_empty_input(self):
        assert sample_clustered([], 10) == []


# ---------------------------------------------------------------------------
# apply_sampling (orchestrator)
# ---------------------------------------------------------------------------


class TestApplySampling:
    def test_mode_all_truncates(self):
        items = make_items(20)
        result = apply_sampling(items, mode="all", limit=5)
        assert len(result) == 5
        # 'all' mode takes first N
        assert [it.id for it in result] == [it.id for it in items[:5]]

    def test_mode_random(self):
        items = make_items(30)
        result = apply_sampling(items, mode="random", limit=10)
        assert len(result) == 10

    def test_mode_stratified(self):
        items = make_items(30)
        result = apply_sampling(items, mode="stratified", limit=10)
        assert len(result) == 10

    def test_unknown_mode_falls_back(self):
        items = make_items(10)
        result = apply_sampling(items, mode="nonexistent", limit=5)
        assert len(result) == 5

    def test_dedup_flag(self):
        items = make_items(5)
        with patch("evalyn_sdk.sampling.deduplicate", return_value=items[:3]) as mock_dedup:
            result = apply_sampling(items, mode="all", limit=10, dedup=True)
            mock_dedup.assert_called_once()
            assert len(result) == 3

    def test_no_dedup_by_default(self):
        items = make_items(5)
        with patch("evalyn_sdk.sampling.deduplicate") as mock_dedup:
            apply_sampling(items, mode="all", limit=10)
            mock_dedup.assert_not_called()

    def test_sampling_modes_constant(self):
        assert "all" in SAMPLING_MODES
        assert "random" in SAMPLING_MODES
        assert "diverse" in SAMPLING_MODES
        assert "stratified" in SAMPLING_MODES
        assert "clustered" in SAMPLING_MODES
