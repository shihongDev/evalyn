"""Tests for gemini_safety module."""

from __future__ import annotations

from datetime import datetime, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.trace.gemini_safety import (
    PROBABILITY_LEVELS,
    SAFETY_CATEGORIES,
    SafetyRating,
    SafetyReport,
    analyze_safety,
    extract_safety_from_span,
    extract_safety_ratings,
    inject_safety_into_span,
    is_safe_response,
)


def _span(sid="s1", **attrs):
    return Span(
        id=sid,
        name="test",
        span_type="llm_call",
        parent_id=None,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# --- SafetyRating as_dict / from_dict ---


def test_safety_rating_as_dict():
    r = SafetyRating(category="HARASSMENT", probability="LOW", blocked=False)
    d = r.as_dict()
    assert d == {"category": "HARASSMENT", "probability": "LOW", "blocked": False}


def test_safety_rating_from_dict():
    d = {"category": "HATE_SPEECH", "probability": "HIGH", "blocked": True}
    r = SafetyRating.from_dict(d)
    assert r.category == "HATE_SPEECH"
    assert r.probability == "HIGH"
    assert r.blocked is True


def test_safety_rating_roundtrip():
    original = SafetyRating(category="DANGEROUS_CONTENT", probability="MEDIUM", blocked=False)
    restored = SafetyRating.from_dict(original.as_dict())
    assert restored.category == original.category
    assert restored.probability == original.probability
    assert restored.blocked == original.blocked


def test_safety_rating_from_dict_defaults():
    r = SafetyRating.from_dict({})
    assert r.category == ""
    assert r.probability == "NEGLIGIBLE"
    assert r.blocked is False


# --- SafetyReport as_dict / from_dict ---


def test_safety_report_as_dict():
    report = SafetyReport(
        ratings=[SafetyRating("HARASSMENT", "LOW")],
        has_block=False,
        overall_safe=True,
        blocked_categories=[],
    )
    d = report.as_dict()
    assert d["has_block"] is False
    assert d["overall_safe"] is True
    assert len(d["ratings"]) == 1
    assert d["blocked_categories"] == []


def test_safety_report_from_dict():
    d = {
        "ratings": [{"category": "HARASSMENT", "probability": "HIGH", "blocked": True}],
        "has_block": True,
        "overall_safe": False,
        "blocked_categories": ["HARASSMENT"],
    }
    report = SafetyReport.from_dict(d)
    assert report.has_block is True
    assert report.overall_safe is False
    assert report.blocked_categories == ["HARASSMENT"]
    assert len(report.ratings) == 1


def test_safety_report_roundtrip():
    original = SafetyReport(
        ratings=[
            SafetyRating("HARASSMENT", "LOW"),
            SafetyRating("HATE_SPEECH", "MEDIUM", blocked=True),
        ],
        has_block=True,
        overall_safe=False,
        blocked_categories=["HATE_SPEECH"],
    )
    restored = SafetyReport.from_dict(original.as_dict())
    assert restored.has_block == original.has_block
    assert restored.overall_safe == original.overall_safe
    assert restored.blocked_categories == original.blocked_categories
    assert len(restored.ratings) == len(original.ratings)


def test_safety_report_format_text_with_ratings():
    report = SafetyReport(
        ratings=[
            SafetyRating("HARASSMENT", "LOW"),
            SafetyRating("HATE_SPEECH", "HIGH", blocked=True),
        ],
        has_block=True,
        overall_safe=False,
        blocked_categories=["HATE_SPEECH"],
    )
    text = report.format_text()
    assert "Safety Report:" in text
    assert "Overall safe: False" in text
    assert "Has block: True" in text
    assert "HATE_SPEECH" in text
    assert "[BLOCKED]" in text


def test_safety_report_format_text_no_ratings():
    report = SafetyReport()
    text = report.format_text()
    assert "No ratings available" in text


# --- extract_safety_ratings ---


def test_extract_safety_ratings_valid():
    response = {
        "candidates": [
            {
                "safetyRatings": [
                    {"category": "HARASSMENT", "probability": "NEGLIGIBLE"},
                    {"category": "HATE_SPEECH", "probability": "LOW"},
                ]
            }
        ]
    }
    ratings = extract_safety_ratings(response)
    assert len(ratings) == 2
    assert ratings[0].category == "HARASSMENT"
    assert ratings[0].probability == "NEGLIGIBLE"
    assert ratings[1].category == "HATE_SPEECH"


def test_extract_safety_ratings_empty_response():
    ratings = extract_safety_ratings({})
    assert ratings == []


def test_extract_safety_ratings_empty_candidates():
    ratings = extract_safety_ratings({"candidates": []})
    assert ratings == []


def test_extract_safety_ratings_no_safety_field():
    ratings = extract_safety_ratings({"candidates": [{"content": "hello"}]})
    assert ratings == []


def test_extract_safety_ratings_with_blocked():
    response = {
        "candidates": [
            {
                "safetyRatings": [
                    {"category": "DANGEROUS_CONTENT", "probability": "HIGH", "blocked": True},
                ]
            }
        ]
    }
    ratings = extract_safety_ratings(response)
    assert len(ratings) == 1
    assert ratings[0].blocked is True


# --- analyze_safety ---


def test_analyze_safety_all_safe():
    ratings = [
        SafetyRating("HARASSMENT", "NEGLIGIBLE"),
        SafetyRating("HATE_SPEECH", "LOW"),
    ]
    report = analyze_safety(ratings)
    assert report.overall_safe is True
    assert report.has_block is False
    assert report.blocked_categories == []


def test_analyze_safety_with_high():
    ratings = [
        SafetyRating("HARASSMENT", "HIGH"),
        SafetyRating("HATE_SPEECH", "LOW"),
    ]
    report = analyze_safety(ratings)
    assert report.overall_safe is False
    assert report.has_block is False


def test_analyze_safety_with_block():
    ratings = [
        SafetyRating("HARASSMENT", "HIGH", blocked=True),
        SafetyRating("HATE_SPEECH", "LOW"),
    ]
    report = analyze_safety(ratings)
    assert report.overall_safe is False
    assert report.has_block is True
    assert report.blocked_categories == ["HARASSMENT"]


def test_analyze_safety_empty():
    report = analyze_safety([])
    assert report.overall_safe is True
    assert report.has_block is False
    assert report.blocked_categories == []


# --- inject / extract span ---


def test_inject_safety_into_span_adds_attribute():
    s = _span()
    ratings = [SafetyRating("HARASSMENT", "LOW")]
    new_s = inject_safety_into_span(s, ratings)
    assert "gemini.safety_ratings" in new_s.attributes
    assert len(new_s.attributes["gemini.safety_ratings"]) == 1


def test_inject_preserves_original_span():
    s = _span(existing="value")
    ratings = [SafetyRating("HARASSMENT", "LOW")]
    new_s = inject_safety_into_span(s, ratings)
    # Original should not be modified
    assert "gemini.safety_ratings" not in s.attributes
    assert s.attributes.get("existing") == "value"
    # New span should have both
    assert "gemini.safety_ratings" in new_s.attributes
    assert new_s.attributes.get("existing") == "value"


def test_extract_safety_from_span_roundtrip():
    s = _span()
    ratings = [
        SafetyRating("HARASSMENT", "LOW"),
        SafetyRating("HATE_SPEECH", "MEDIUM"),
    ]
    new_s = inject_safety_into_span(s, ratings)
    report = extract_safety_from_span(new_s)
    assert report is not None
    assert len(report.ratings) == 2
    assert report.overall_safe is True


def test_extract_safety_from_span_none():
    s = _span()
    result = extract_safety_from_span(s)
    assert result is None


# --- is_safe_response ---


def test_is_safe_negligible():
    ratings = [SafetyRating("HARASSMENT", "NEGLIGIBLE")]
    assert is_safe_response(ratings) is True


def test_is_safe_low():
    ratings = [SafetyRating("HARASSMENT", "LOW")]
    assert is_safe_response(ratings) is True


def test_is_safe_medium():
    ratings = [SafetyRating("HARASSMENT", "MEDIUM")]
    assert is_safe_response(ratings) is True


def test_is_safe_high_exceeds_medium():
    ratings = [SafetyRating("HARASSMENT", "HIGH")]
    assert is_safe_response(ratings) is False


def test_is_safe_high_with_max_high():
    ratings = [SafetyRating("HARASSMENT", "HIGH")]
    assert is_safe_response(ratings, max_probability="HIGH") is True


def test_is_safe_blocked():
    ratings = [SafetyRating("HARASSMENT", "LOW", blocked=True)]
    assert is_safe_response(ratings) is False


def test_is_safe_low_max_negligible():
    ratings = [SafetyRating("HARASSMENT", "LOW")]
    assert is_safe_response(ratings, max_probability="NEGLIGIBLE") is False


def test_is_safe_empty():
    assert is_safe_response([]) is True


# --- Constants ---


def test_safety_categories():
    assert "HARASSMENT" in SAFETY_CATEGORIES
    assert "HATE_SPEECH" in SAFETY_CATEGORIES
    assert "SEXUALLY_EXPLICIT" in SAFETY_CATEGORIES
    assert "DANGEROUS_CONTENT" in SAFETY_CATEGORIES
    assert len(SAFETY_CATEGORIES) == 4


def test_probability_levels():
    assert PROBABILITY_LEVELS == ["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"]
