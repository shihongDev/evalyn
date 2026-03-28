from __future__ import annotations

import os

import pytest

from evalyn_sdk.metrics.image_evaluation import (
    ImageEvalReport,
    ImageMetadata,
    ImageScore,
    check_image_dimensions,
    check_image_exists,
    check_image_format,
    check_image_size,
    evaluate_image,
    extract_image_metadata_from_path,
)


# ---------------------------------------------------------------------------
# ImageMetadata
# ---------------------------------------------------------------------------


class TestImageMetadata:
    def test_defaults(self):
        m = ImageMetadata()
        assert m.width == 0
        assert m.height == 0
        assert m.format == ""
        assert m.size_bytes == 0
        assert m.has_content is False

    def test_as_dict(self):
        m = ImageMetadata(width=800, height=600, format="png", size_bytes=1024, has_content=True)
        d = m.as_dict()
        assert d["width"] == 800
        assert d["height"] == 600
        assert d["format"] == "png"
        assert d["size_bytes"] == 1024
        assert d["has_content"] is True

    def test_from_dict_roundtrip(self):
        m = ImageMetadata(width=100, height=200, format="jpg", size_bytes=500, has_content=True)
        restored = ImageMetadata.from_dict(m.as_dict())
        assert restored == m

    def test_from_dict_missing_keys(self):
        m = ImageMetadata.from_dict({})
        assert m.width == 0
        assert m.format == ""


# ---------------------------------------------------------------------------
# ImageScore
# ---------------------------------------------------------------------------


class TestImageScore:
    def test_as_dict(self):
        s = ImageScore(metric_name="test", score=0.5, passed=False, details={"a": 1})
        d = s.as_dict()
        assert d["metric_name"] == "test"
        assert d["score"] == 0.5
        assert d["passed"] is False
        assert d["details"] == {"a": 1}

    def test_default_details(self):
        s = ImageScore(metric_name="x", score=1.0, passed=True)
        assert s.details == {}


# ---------------------------------------------------------------------------
# ImageEvalReport
# ---------------------------------------------------------------------------


class TestImageEvalReport:
    def test_defaults(self):
        r = ImageEvalReport()
        assert r.scores == []
        assert r.overall_score == 0.0
        assert r.pass_rate == 0.0

    def test_as_dict_from_dict_roundtrip(self):
        scores = [
            ImageScore("a", 1.0, True, {"k": "v"}),
            ImageScore("b", 0.0, False),
        ]
        r = ImageEvalReport(scores=scores, overall_score=0.5, pass_rate=0.5)
        d = r.as_dict()
        restored = ImageEvalReport.from_dict(d)
        assert len(restored.scores) == 2
        assert restored.scores[0].metric_name == "a"
        assert restored.scores[1].passed is False
        assert restored.overall_score == 0.5
        assert restored.pass_rate == 0.5

    def test_format_text_contains_header(self):
        r = ImageEvalReport(
            scores=[ImageScore("check", 1.0, True)],
            overall_score=1.0,
            pass_rate=1.0,
        )
        text = r.format_text()
        assert "Image Evaluation Report" in text
        assert "PASS" in text
        assert "1.00" in text

    def test_format_text_fail(self):
        r = ImageEvalReport(
            scores=[ImageScore("check", 0.0, False)],
            overall_score=0.0,
            pass_rate=0.0,
        )
        text = r.format_text()
        assert "FAIL" in text


# ---------------------------------------------------------------------------
# check_image_exists
# ---------------------------------------------------------------------------


class TestCheckImageExists:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG fake content")
        result = check_image_exists(str(f))
        assert result.passed is True
        assert result.score == 1.0

    def test_missing_file(self, tmp_path):
        result = check_image_exists(str(tmp_path / "nope.png"))
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.png"
        f.write_bytes(b"")
        result = check_image_exists(str(f))
        assert result.passed is False
        assert "empty" in result.details.get("reason", "")

    def test_empty_path(self):
        result = check_image_exists("")
        assert result.passed is False
        assert "empty path" in result.details.get("reason", "")


# ---------------------------------------------------------------------------
# check_image_format
# ---------------------------------------------------------------------------


class TestCheckImageFormat:
    def test_correct_format(self):
        result = check_image_format("/some/image.png")
        assert result.passed is True

    def test_wrong_format(self):
        result = check_image_format("/some/image.bmp")
        assert result.passed is False

    def test_custom_expected(self):
        result = check_image_format("/file.bmp", expected_formats=["bmp", "tiff"])
        assert result.passed is True

    def test_case_insensitive(self):
        result = check_image_format("/file.PNG")
        assert result.passed is True

    def test_no_extension(self):
        result = check_image_format("/noext")
        assert result.passed is False


# ---------------------------------------------------------------------------
# check_image_size
# ---------------------------------------------------------------------------


class TestCheckImageSize:
    def test_within_bounds(self, tmp_path):
        f = tmp_path / "ok.png"
        f.write_bytes(b"x" * 500)
        result = check_image_size(str(f))
        assert result.passed is True

    def test_too_small(self, tmp_path):
        f = tmp_path / "tiny.png"
        f.write_bytes(b"x" * 10)
        result = check_image_size(str(f), min_bytes=100)
        assert result.passed is False

    def test_too_large(self, tmp_path):
        f = tmp_path / "big.png"
        f.write_bytes(b"x" * 200)
        result = check_image_size(str(f), max_bytes=100)
        assert result.passed is False

    def test_missing_file(self, tmp_path):
        result = check_image_size(str(tmp_path / "gone.png"))
        assert result.passed is False


# ---------------------------------------------------------------------------
# check_image_dimensions
# ---------------------------------------------------------------------------


class TestCheckImageDimensions:
    def test_valid_dimensions(self):
        m = ImageMetadata(width=800, height=600)
        result = check_image_dimensions(m)
        assert result.passed is True

    def test_too_wide(self):
        m = ImageMetadata(width=20000, height=100)
        result = check_image_dimensions(m, max_width=10000)
        assert result.passed is False

    def test_too_tall(self):
        m = ImageMetadata(width=100, height=20000)
        result = check_image_dimensions(m, max_height=10000)
        assert result.passed is False

    def test_zero_dimensions(self):
        m = ImageMetadata(width=0, height=0)
        result = check_image_dimensions(m, min_width=1, min_height=1)
        assert result.passed is False

    def test_exact_bounds(self):
        m = ImageMetadata(width=100, height=200)
        result = check_image_dimensions(m, min_width=100, min_height=200, max_width=100, max_height=200)
        assert result.passed is True


# ---------------------------------------------------------------------------
# extract_image_metadata_from_path
# ---------------------------------------------------------------------------


class TestExtractImageMetadata:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "pic.jpeg"
        f.write_bytes(b"data" * 50)
        m = extract_image_metadata_from_path(str(f))
        assert m.format == "jpeg"
        assert m.size_bytes == 200
        assert m.has_content is True

    def test_missing_file(self):
        m = extract_image_metadata_from_path("/does/not/exist.webp")
        assert m.format == "webp"
        assert m.size_bytes == 0
        assert m.has_content is False

    def test_no_extension(self):
        m = extract_image_metadata_from_path("/some/file")
        assert m.format == ""


# ---------------------------------------------------------------------------
# evaluate_image
# ---------------------------------------------------------------------------


class TestEvaluateImage:
    def test_full_pipeline_existing(self, tmp_path):
        f = tmp_path / "good.png"
        f.write_bytes(b"x" * 500)
        report = evaluate_image(str(f))
        assert len(report.scores) == 3  # exists, format, size (no dims since w/h=0)
        assert report.pass_rate == 1.0

    def test_full_pipeline_with_metadata(self, tmp_path):
        f = tmp_path / "good.png"
        f.write_bytes(b"x" * 500)
        meta = ImageMetadata(width=640, height=480)
        report = evaluate_image(str(f), metadata=meta)
        assert len(report.scores) == 4  # exists, format, size, dimensions
        assert report.pass_rate == 1.0

    def test_missing_file_pipeline(self):
        report = evaluate_image("/missing/file.png")
        assert report.pass_rate < 1.0
        # exists and size fail, format passes (extension is png)
        assert any(s.metric_name == "image_exists" and not s.passed for s in report.scores)

    def test_overall_score_computed(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_bytes(b"x" * 500)
        report = evaluate_image(str(f))
        # All pass: overall should be 1.0
        assert report.overall_score == pytest.approx(1.0)
