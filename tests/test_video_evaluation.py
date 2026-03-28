from __future__ import annotations

import os

import pytest

from evalyn_sdk.metrics.video_evaluation import (
    VideoEvalReport,
    VideoMetadata,
    VideoScore,
    check_video_duration,
    check_video_exists,
    check_video_format,
    check_video_resolution,
    evaluate_video,
    extract_video_metadata_from_path,
)


# ---------------------------------------------------------------------------
# VideoMetadata
# ---------------------------------------------------------------------------


class TestVideoMetadata:
    def test_defaults(self):
        m = VideoMetadata()
        assert m.duration_seconds == 0.0
        assert m.width == 0
        assert m.height == 0
        assert m.fps == 0.0
        assert m.format == ""
        assert m.size_bytes == 0

    def test_as_dict(self):
        m = VideoMetadata(
            duration_seconds=10.5, width=1920, height=1080, fps=30.0,
            format="mp4", size_bytes=2048,
        )
        d = m.as_dict()
        assert d["duration_seconds"] == 10.5
        assert d["width"] == 1920
        assert d["height"] == 1080
        assert d["fps"] == 30.0
        assert d["format"] == "mp4"
        assert d["size_bytes"] == 2048

    def test_from_dict_roundtrip(self):
        m = VideoMetadata(
            duration_seconds=5.0, width=640, height=480, fps=24.0,
            format="avi", size_bytes=999,
        )
        restored = VideoMetadata.from_dict(m.as_dict())
        assert restored == m

    def test_from_dict_missing_keys(self):
        m = VideoMetadata.from_dict({})
        assert m.duration_seconds == 0.0
        assert m.width == 0
        assert m.format == ""


# ---------------------------------------------------------------------------
# VideoScore
# ---------------------------------------------------------------------------


class TestVideoScore:
    def test_as_dict(self):
        s = VideoScore(metric_name="test", score=0.5, passed=False, details={"a": 1})
        d = s.as_dict()
        assert d["metric_name"] == "test"
        assert d["score"] == 0.5
        assert d["passed"] is False
        assert d["details"] == {"a": 1}

    def test_default_details(self):
        s = VideoScore(metric_name="x", score=1.0, passed=True)
        assert s.details == {}


# ---------------------------------------------------------------------------
# VideoEvalReport
# ---------------------------------------------------------------------------


class TestVideoEvalReport:
    def test_defaults(self):
        r = VideoEvalReport()
        assert r.scores == []
        assert r.overall_score == 0.0
        assert r.pass_rate == 0.0

    def test_as_dict_from_dict_roundtrip(self):
        scores = [
            VideoScore("a", 1.0, True, {"k": "v"}),
            VideoScore("b", 0.0, False),
        ]
        r = VideoEvalReport(scores=scores, overall_score=0.5, pass_rate=0.5)
        d = r.as_dict()
        restored = VideoEvalReport.from_dict(d)
        assert len(restored.scores) == 2
        assert restored.scores[0].metric_name == "a"
        assert restored.scores[1].passed is False
        assert restored.overall_score == 0.5
        assert restored.pass_rate == 0.5

    def test_format_text_contains_header(self):
        r = VideoEvalReport(
            scores=[VideoScore("check", 1.0, True)],
            overall_score=1.0,
            pass_rate=1.0,
        )
        text = r.format_text()
        assert "Video Evaluation Report" in text
        assert "PASS" in text
        assert "1.00" in text

    def test_format_text_fail(self):
        r = VideoEvalReport(
            scores=[VideoScore("check", 0.0, False)],
            overall_score=0.0,
            pass_rate=0.0,
        )
        text = r.format_text()
        assert "FAIL" in text

    def test_format_text_pass_rate(self):
        r = VideoEvalReport(
            scores=[VideoScore("a", 1.0, True), VideoScore("b", 0.0, False)],
            overall_score=0.5,
            pass_rate=0.5,
        )
        text = r.format_text()
        assert "50.0%" in text


# ---------------------------------------------------------------------------
# check_video_exists
# ---------------------------------------------------------------------------


class TestCheckVideoExists:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"\x00\x00\x00\x1cftypisom")
        result = check_video_exists(str(f))
        assert result.passed is True
        assert result.score == 1.0

    def test_missing_file(self, tmp_path):
        result = check_video_exists(str(tmp_path / "nope.mp4"))
        assert result.passed is False
        assert result.score == 0.0

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.mp4"
        f.write_bytes(b"")
        result = check_video_exists(str(f))
        assert result.passed is False
        assert "empty" in result.details.get("reason", "")

    def test_empty_path(self):
        result = check_video_exists("")
        assert result.passed is False
        assert "empty path" in result.details.get("reason", "")


# ---------------------------------------------------------------------------
# check_video_format
# ---------------------------------------------------------------------------


class TestCheckVideoFormat:
    def test_correct_format_mp4(self):
        result = check_video_format("/some/video.mp4")
        assert result.passed is True

    def test_correct_format_webm(self):
        result = check_video_format("/some/video.webm")
        assert result.passed is True

    def test_correct_format_avi(self):
        result = check_video_format("/some/video.avi")
        assert result.passed is True

    def test_correct_format_mov(self):
        result = check_video_format("/some/video.mov")
        assert result.passed is True

    def test_wrong_format(self):
        result = check_video_format("/some/video.flv")
        assert result.passed is False

    def test_custom_expected(self):
        result = check_video_format("/file.flv", expected_formats=["flv", "mkv"])
        assert result.passed is True

    def test_case_insensitive(self):
        result = check_video_format("/file.MP4")
        assert result.passed is True

    def test_no_extension(self):
        result = check_video_format("/noext")
        assert result.passed is False


# ---------------------------------------------------------------------------
# check_video_duration
# ---------------------------------------------------------------------------


class TestCheckVideoDuration:
    def test_within_bounds(self):
        m = VideoMetadata(duration_seconds=30.0)
        result = check_video_duration(m)
        assert result.passed is True

    def test_too_short(self):
        m = VideoMetadata(duration_seconds=0.01)
        result = check_video_duration(m, min_seconds=0.1)
        assert result.passed is False

    def test_too_long(self):
        m = VideoMetadata(duration_seconds=700.0)
        result = check_video_duration(m, max_seconds=600.0)
        assert result.passed is False

    def test_exact_min_bound(self):
        m = VideoMetadata(duration_seconds=0.1)
        result = check_video_duration(m, min_seconds=0.1)
        assert result.passed is True

    def test_exact_max_bound(self):
        m = VideoMetadata(duration_seconds=600.0)
        result = check_video_duration(m, max_seconds=600.0)
        assert result.passed is True

    def test_zero_duration(self):
        m = VideoMetadata(duration_seconds=0.0)
        result = check_video_duration(m, min_seconds=0.1)
        assert result.passed is False


# ---------------------------------------------------------------------------
# check_video_resolution
# ---------------------------------------------------------------------------


class TestCheckVideoResolution:
    def test_valid_resolution(self):
        m = VideoMetadata(width=1920, height=1080)
        result = check_video_resolution(m)
        assert result.passed is True

    def test_too_small_width(self):
        m = VideoMetadata(width=0, height=1080)
        result = check_video_resolution(m, min_width=1)
        assert result.passed is False

    def test_too_small_height(self):
        m = VideoMetadata(width=1920, height=0)
        result = check_video_resolution(m, min_height=1)
        assert result.passed is False

    def test_exact_min_bound(self):
        m = VideoMetadata(width=720, height=480)
        result = check_video_resolution(m, min_width=720, min_height=480)
        assert result.passed is True

    def test_zero_dimensions(self):
        m = VideoMetadata(width=0, height=0)
        result = check_video_resolution(m, min_width=1, min_height=1)
        assert result.passed is False


# ---------------------------------------------------------------------------
# extract_video_metadata_from_path
# ---------------------------------------------------------------------------


class TestExtractVideoMetadata:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"data" * 50)
        m = extract_video_metadata_from_path(str(f))
        assert m.format == "mp4"
        assert m.size_bytes == 200

    def test_missing_file(self):
        m = extract_video_metadata_from_path("/does/not/exist.webm")
        assert m.format == "webm"
        assert m.size_bytes == 0

    def test_no_extension(self):
        m = extract_video_metadata_from_path("/some/file")
        assert m.format == ""


# ---------------------------------------------------------------------------
# evaluate_video
# ---------------------------------------------------------------------------


class TestEvaluateVideo:
    def test_full_pipeline_existing(self, tmp_path):
        f = tmp_path / "good.mp4"
        f.write_bytes(b"x" * 500)
        report = evaluate_video(str(f))
        # exists + format (no duration or resolution since metadata has zeros)
        assert len(report.scores) == 2
        assert report.pass_rate == 1.0

    def test_full_pipeline_with_metadata(self, tmp_path):
        f = tmp_path / "good.mp4"
        f.write_bytes(b"x" * 500)
        meta = VideoMetadata(duration_seconds=10.0, width=1920, height=1080)
        report = evaluate_video(str(f), metadata=meta)
        # exists + format + duration + resolution
        assert len(report.scores) == 4
        assert report.pass_rate == 1.0

    def test_missing_file_pipeline(self):
        report = evaluate_video("/missing/file.mp4")
        assert report.pass_rate < 1.0
        assert any(
            s.metric_name == "video_exists" and not s.passed for s in report.scores
        )

    def test_overall_score_computed(self, tmp_path):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"x" * 500)
        report = evaluate_video(str(f))
        assert report.overall_score == pytest.approx(1.0)

    def test_evaluate_with_duration_only(self, tmp_path):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"x" * 100)
        meta = VideoMetadata(duration_seconds=5.0)
        report = evaluate_video(str(f), metadata=meta)
        # exists + format + duration (no resolution since w/h are 0)
        assert len(report.scores) == 3

    def test_evaluate_wrong_format(self, tmp_path):
        f = tmp_path / "test.flv"
        f.write_bytes(b"x" * 100)
        report = evaluate_video(str(f))
        format_score = next(s for s in report.scores if s.metric_name == "video_format")
        assert format_score.passed is False
