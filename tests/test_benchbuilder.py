"""Tests for BenchBuilder auto-curation module."""

from __future__ import annotations

import pytest

from evalyn_sdk.benchbuilder import (
    CurationConfig,
    CurationResult,
    TraceScore,
    cluster_traces,
    curate_benchmark,
    format_curation_report,
    score_trace_difficulty,
    score_trace_quality,
)


# ---------------------------------------------------------------------------
# TraceScore dataclass
# ---------------------------------------------------------------------------


class TestTraceScore:
    def test_basic_creation(self):
        ts = TraceScore(trace_id="t1", quality_score=0.8, difficulty_score=0.5)
        assert ts.trace_id == "t1"
        assert ts.quality_score == 0.8
        assert ts.difficulty_score == 0.5
        assert ts.topic_cluster == -1

    def test_custom_cluster(self):
        ts = TraceScore(trace_id="t1", quality_score=0.5, difficulty_score=0.5, topic_cluster=3)
        assert ts.topic_cluster == 3

    def test_as_dict(self):
        ts = TraceScore(trace_id="t1", quality_score=0.7, difficulty_score=0.3, topic_cluster=2)
        d = ts.as_dict()
        assert d["trace_id"] == "t1"
        assert d["quality_score"] == 0.7
        assert d["topic_cluster"] == 2

    def test_from_dict(self):
        data = {"trace_id": "t2", "quality_score": 0.9, "difficulty_score": 0.1, "topic_cluster": 1}
        ts = TraceScore.from_dict(data)
        assert ts.trace_id == "t2"
        assert ts.quality_score == 0.9

    def test_from_dict_default_cluster(self):
        data = {"trace_id": "t3", "quality_score": 0.5, "difficulty_score": 0.5}
        ts = TraceScore.from_dict(data)
        assert ts.topic_cluster == -1

    def test_roundtrip(self):
        original = TraceScore(trace_id="t1", quality_score=0.6, difficulty_score=0.4, topic_cluster=2)
        restored = TraceScore.from_dict(original.as_dict())
        assert restored.trace_id == original.trace_id
        assert restored.quality_score == original.quality_score
        assert restored.difficulty_score == original.difficulty_score
        assert restored.topic_cluster == original.topic_cluster


# ---------------------------------------------------------------------------
# CurationConfig dataclass
# ---------------------------------------------------------------------------


class TestCurationConfig:
    def test_defaults(self):
        cfg = CurationConfig()
        assert cfg.target_size == 100
        assert cfg.min_quality == 0.3
        assert cfg.diversity_weight == 0.5
        assert cfg.seed is None

    def test_custom(self):
        cfg = CurationConfig(target_size=50, min_quality=0.5, diversity_weight=0.8, seed=42)
        assert cfg.target_size == 50
        assert cfg.seed == 42

    def test_as_dict(self):
        cfg = CurationConfig(target_size=30, seed=7)
        d = cfg.as_dict()
        assert d["target_size"] == 30
        assert d["seed"] == 7

    def test_from_dict(self):
        data = {"target_size": 20, "min_quality": 0.1, "diversity_weight": 0.9, "seed": 99}
        cfg = CurationConfig.from_dict(data)
        assert cfg.target_size == 20
        assert cfg.min_quality == 0.1

    def test_from_dict_defaults(self):
        cfg = CurationConfig.from_dict({})
        assert cfg.target_size == 100
        assert cfg.seed is None

    def test_roundtrip(self):
        original = CurationConfig(target_size=42, min_quality=0.2, diversity_weight=0.7, seed=123)
        restored = CurationConfig.from_dict(original.as_dict())
        assert restored.target_size == original.target_size
        assert restored.min_quality == original.min_quality
        assert restored.seed == original.seed


# ---------------------------------------------------------------------------
# CurationResult dataclass
# ---------------------------------------------------------------------------


class TestCurationResult:
    def test_empty(self):
        r = CurationResult()
        assert r.selected_ids == []
        assert r.scores == []
        assert r.total_traces == 0

    def test_as_dict(self):
        scores = [TraceScore(trace_id="t1", quality_score=0.8, difficulty_score=0.5)]
        r = CurationResult(
            selected_ids=["t1"],
            scores=scores,
            total_traces=10,
            avg_quality=0.8,
            avg_difficulty=0.5,
            n_clusters_represented=3,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["t1"]
        assert len(d["scores"]) == 1
        assert d["n_clusters_represented"] == 3

    def test_from_dict(self):
        data = {
            "selected_ids": ["t1", "t2"],
            "scores": [
                {"trace_id": "t1", "quality_score": 0.8, "difficulty_score": 0.5},
            ],
            "total_traces": 5,
            "avg_quality": 0.8,
            "avg_difficulty": 0.5,
            "n_clusters_represented": 2,
        }
        r = CurationResult.from_dict(data)
        assert len(r.selected_ids) == 2
        assert len(r.scores) == 1

    def test_roundtrip(self):
        scores = [TraceScore(trace_id="t1", quality_score=0.7, difficulty_score=0.3, topic_cluster=1)]
        original = CurationResult(
            selected_ids=["t1"],
            scores=scores,
            total_traces=5,
            avg_quality=0.7,
            avg_difficulty=0.3,
            n_clusters_represented=1,
        )
        restored = CurationResult.from_dict(original.as_dict())
        assert restored.selected_ids == original.selected_ids
        assert restored.total_traces == original.total_traces


# ---------------------------------------------------------------------------
# Quality Scoring
# ---------------------------------------------------------------------------


class TestScoreTraceQuality:
    def test_empty_string(self):
        assert score_trace_quality("") == 0.0

    def test_whitespace_only(self):
        assert score_trace_quality("   ") == 0.0

    def test_very_short_text(self):
        score = score_trace_quality("hi")
        assert 0.0 <= score < 0.7  # Short text penalized relative to longer text

    def test_reasonable_length(self):
        text = "Please explain the concept of machine learning. " \
               "Include examples of supervised and unsupervised approaches. " \
               "Discuss the advantages and limitations of each method."
        score = score_trace_quality(text)
        assert score > 0.3

    def test_long_text(self):
        text = " ".join(["word"] * 600)
        score = score_trace_quality(text)
        assert 0.0 <= score <= 1.0

    def test_high_diversity(self):
        # All unique words -> high diversity
        words = [f"word{i}" for i in range(50)]
        text = " ".join(words) + ". " + " ".join(words[25:]) + "."
        score = score_trace_quality(text)
        assert score > 0.3

    def test_low_diversity(self):
        # Same word repeated -> low diversity
        text = " ".join(["same"] * 50) + "."
        score = score_trace_quality(text)
        # Still gets some score from length/structure
        assert 0.0 <= score <= 1.0

    def test_multiple_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        score = score_trace_quality(text)
        assert score > 0.0

    def test_returns_zero_to_one(self):
        texts = [
            "short",
            "A longer sentence with more words.",
            "Very " * 100 + "long text.",
        ]
        for t in texts:
            s = score_trace_quality(t)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Difficulty Scoring
# ---------------------------------------------------------------------------


class TestScoreTraceDifficulty:
    def test_empty_string(self):
        assert score_trace_difficulty("") == 0.0

    def test_whitespace_only(self):
        assert score_trace_difficulty("   ") == 0.0

    def test_short_simple(self):
        score = score_trace_difficulty("hello world")
        assert score < 0.5

    def test_long_complex(self):
        text = (
            "Analyze the epistemological implications of computational "
            "neuroscience methodologies. Compare and contrast the theoretical "
            "frameworks underlying connectionist architectures with classical "
            "symbolic approaches. Evaluate the philosophical ramifications "
            "of each paradigm for understanding consciousness."
        )
        score = score_trace_difficulty(text)
        assert score > 0.3

    def test_question_words_increase_difficulty(self):
        simple = "Tell me about cats."
        complex_q = "Explain why cats evolved differently. Analyze how their behavior compares to dogs."
        simple_score = score_trace_difficulty(simple)
        complex_score = score_trace_difficulty(complex_q)
        assert complex_score >= simple_score

    def test_returns_zero_to_one(self):
        texts = [
            "hi",
            "Explain the concept of recursion.",
            "Why " * 100 + "is this important?",
        ]
        for t in texts:
            s = score_trace_difficulty(t)
            assert 0.0 <= s <= 1.0

    def test_longer_words_increase_difficulty(self):
        short_words = "I like to run and jump and play and eat."
        long_words = "Epistemological methodologies necessitate comprehensive philosophical investigations."
        assert score_trace_difficulty(long_words) > score_trace_difficulty(short_words)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestClusterTraces:
    def test_empty(self):
        result = cluster_traces({})
        assert result == {}

    def test_single_trace(self):
        result = cluster_traces({"t1": "hello world"})
        assert "t1" in result
        assert isinstance(result["t1"], int)

    def test_all_assigned(self):
        texts = {f"t{i}": f"text about topic number {i}" for i in range(10)}
        result = cluster_traces(texts, n_clusters=3)
        assert set(result.keys()) == set(texts.keys())

    def test_cluster_range(self):
        texts = {f"t{i}": f"different topic {i}" for i in range(10)}
        result = cluster_traces(texts, n_clusters=3)
        cluster_ids = set(result.values())
        for c in cluster_ids:
            assert 0 <= c < 3

    def test_similar_texts_same_cluster(self):
        texts = {
            "t1": "machine learning algorithms neural networks deep learning",
            "t2": "machine learning models neural network training",
            "t3": "cooking recipes baking bread sourdough yeast",
            "t4": "cooking techniques recipe preparation kitchen tools",
        }
        result = cluster_traces(texts, n_clusters=2)
        # ML texts should be in same cluster
        assert result["t1"] == result["t2"]
        # Cooking texts should be in same cluster
        assert result["t3"] == result["t4"]

    def test_n_clusters_clamped(self):
        texts = {"t1": "hello", "t2": "world"}
        result = cluster_traces(texts, n_clusters=10)
        assert len(result) == 2

    def test_one_cluster(self):
        texts = {f"t{i}": f"text {i}" for i in range(5)}
        result = cluster_traces(texts, n_clusters=1)
        assert all(v == 0 for v in result.values())


# ---------------------------------------------------------------------------
# Curation Pipeline
# ---------------------------------------------------------------------------


def _make_traces(n: int) -> dict[str, str]:
    """Generate diverse test traces."""
    topics = [
        "Explain machine learning algorithms and neural network architectures. "
        "Discuss supervised and unsupervised learning approaches.",
        "Describe the process of photosynthesis in plants. "
        "Include the light-dependent and light-independent reactions.",
        "Analyze the economic factors that led to the industrial revolution. "
        "Compare the economic conditions in different European countries.",
        "Discuss the principles of quantum mechanics and wave-particle duality. "
        "Explain the uncertainty principle and its implications.",
        "Evaluate the effectiveness of different programming paradigms. "
        "Compare object-oriented and functional programming approaches.",
    ]
    traces = {}
    for i in range(n):
        topic = topics[i % len(topics)]
        traces[f"trace_{i}"] = topic + f" Variation number {i}."
    return traces


class TestCurateBenchmark:
    def test_empty_traces(self):
        result = curate_benchmark({})
        assert result.total_traces == 0
        assert result.selected_ids == []

    def test_basic_curation(self):
        traces = _make_traces(20)
        config = CurationConfig(target_size=10, seed=42)
        result = curate_benchmark(traces, config)
        assert len(result.selected_ids) <= 10
        assert result.total_traces == 20

    def test_all_selected_when_target_large(self):
        traces = _make_traces(5)
        config = CurationConfig(target_size=100, min_quality=0.0, seed=42)
        result = curate_benchmark(traces, config)
        assert len(result.selected_ids) == 5

    def test_min_quality_filter(self):
        traces = {"t1": "x", "t2": "A reasonable text about machine learning concepts. It has multiple sentences."}
        config = CurationConfig(target_size=10, min_quality=0.5, seed=42)
        result = curate_benchmark(traces, config)
        # Very short text should be filtered
        for tid in result.selected_ids:
            score_map = {s.trace_id: s for s in result.scores}
            assert score_map[tid].quality_score >= 0.5

    def test_scores_computed(self):
        traces = _make_traces(10)
        result = curate_benchmark(traces)
        assert len(result.scores) == 10
        for s in result.scores:
            assert 0.0 <= s.quality_score <= 1.0
            assert 0.0 <= s.difficulty_score <= 1.0

    def test_avg_quality(self):
        traces = _make_traces(10)
        config = CurationConfig(target_size=5, seed=42)
        result = curate_benchmark(traces, config)
        if result.selected_ids:
            score_map = {s.trace_id: s for s in result.scores}
            expected_avg = sum(
                score_map[tid].quality_score for tid in result.selected_ids
            ) / len(result.selected_ids)
            assert abs(result.avg_quality - expected_avg) < 1e-10

    def test_avg_difficulty(self):
        traces = _make_traces(10)
        config = CurationConfig(target_size=5, seed=42)
        result = curate_benchmark(traces, config)
        if result.selected_ids:
            score_map = {s.trace_id: s for s in result.scores}
            expected_avg = sum(
                score_map[tid].difficulty_score for tid in result.selected_ids
            ) / len(result.selected_ids)
            assert abs(result.avg_difficulty - expected_avg) < 1e-10

    def test_clusters_represented(self):
        traces = _make_traces(20)
        config = CurationConfig(target_size=10, seed=42)
        result = curate_benchmark(traces, config)
        assert result.n_clusters_represented >= 1

    def test_no_duplicates(self):
        traces = _make_traces(20)
        config = CurationConfig(target_size=10, seed=42)
        result = curate_benchmark(traces, config)
        assert len(result.selected_ids) == len(set(result.selected_ids))

    def test_deterministic_with_seed(self):
        traces = _make_traces(20)
        config = CurationConfig(target_size=10, seed=42)
        r1 = curate_benchmark(traces, config)
        r2 = curate_benchmark(traces, config)
        assert r1.selected_ids == r2.selected_ids

    def test_default_config(self):
        traces = _make_traces(10)
        result = curate_benchmark(traces)
        assert result.total_traces == 10

    def test_all_below_min_quality(self):
        # Very short traces that will score low
        traces = {"t1": "x", "t2": "y", "t3": "z"}
        config = CurationConfig(min_quality=0.99, seed=42)
        result = curate_benchmark(traces, config)
        assert result.selected_ids == []
        assert result.total_traces == 3

    def test_result_as_dict_roundtrip(self):
        traces = _make_traces(10)
        config = CurationConfig(target_size=5, seed=42)
        result = curate_benchmark(traces, config)
        restored = CurationResult.from_dict(result.as_dict())
        assert restored.selected_ids == result.selected_ids
        assert restored.total_traces == result.total_traces

    def test_cluster_diversity(self):
        # Traces from clearly different topics should produce multiple clusters
        traces = {
            "ml_1": "Machine learning neural networks deep learning algorithms training models.",
            "ml_2": "Supervised learning classification regression neural network architectures.",
            "cook_1": "Cooking recipes baking bread sourdough techniques kitchen preparation.",
            "cook_2": "Culinary arts recipe development food preparation cooking methods.",
            "astro_1": "Astronomy telescopes galaxies stars planetary nebula observations.",
            "astro_2": "Astrophysics cosmology dark matter gravitational waves observations.",
        }
        config = CurationConfig(target_size=6, min_quality=0.0, seed=42)
        result = curate_benchmark(traces, config)
        assert result.n_clusters_represented >= 2


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestFormatCurationReport:
    def test_empty_result(self):
        result = CurationResult()
        report = format_curation_report(result)
        assert "BenchBuilder Curation Report" in report
        assert "Total traces evaluated: 0" in report

    def test_with_selections(self):
        scores = [
            TraceScore(trace_id="t1", quality_score=0.8, difficulty_score=0.5, topic_cluster=0),
            TraceScore(trace_id="t2", quality_score=0.7, difficulty_score=0.6, topic_cluster=1),
        ]
        result = CurationResult(
            selected_ids=["t1", "t2"],
            scores=scores,
            total_traces=10,
            avg_quality=0.75,
            avg_difficulty=0.55,
            n_clusters_represented=2,
        )
        report = format_curation_report(result)
        assert "t1" in report
        assert "t2" in report
        assert "Selected for benchmark: 2" in report
        assert "Clusters represented: 2" in report

    def test_report_is_string(self):
        result = CurationResult(total_traces=5)
        assert isinstance(format_curation_report(result), str)

    def test_contains_header(self):
        result = CurationResult()
        report = format_curation_report(result)
        assert "=" in report
