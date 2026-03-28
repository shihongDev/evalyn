"""Tests for evalyn_sdk.metrics.audio_evaluation module."""

from __future__ import annotations

import os
import tempfile

import pytest

from evalyn_sdk.metrics.audio_evaluation import (
    AudioEvalReport,
    AudioMetadata,
    AudioScore,
    check_audio_duration,
    check_audio_exists,
    check_audio_format,
    check_audio_size,
    evaluate_audio,
    extract_audio_metadata_from_path,
)


# ---------------------------------------------------------------------------
# AudioMetadata
# ---------------------------------------------------------------------------


class TestAudioMetadata:
    def test_as_dict(self):
        m = AudioMetadata(duration_seconds=5.0, sample_rate=44100, channels=2, format="wav", size_bytes=1000)
        d = m.as_dict()
        assert d["duration_seconds"] == 5.0
        assert d["sample_rate"] == 44100
        assert d["channels"] == 2
        assert d["format"] == "wav"
        assert d["size_bytes"] == 1000

    def test_from_dict(self):
        m = AudioMetadata.from_dict({"duration_seconds": 3.5, "format": "mp3"})
        assert m.duration_seconds == 3.5
        assert m.format == "mp3"
        assert m.sample_rate == 0

    def test_roundtrip(self):
        original = AudioMetadata(duration_seconds=10.0, sample_rate=48000, channels=1, format="flac", size_bytes=50000)
        restored = AudioMetadata.from_dict(original.as_dict())
        assert restored.duration_seconds == original.duration_seconds
        assert restored.sample_rate == original.sample_rate
        assert restored.channels == original.channels
        assert restored.format == original.format
        assert restored.size_bytes == original.size_bytes

    def test_defaults(self):
        m = AudioMetadata()
        assert m.duration_seconds == 0.0
        assert m.sample_rate == 0
        assert m.channels == 0
        assert m.format == ""
        assert m.size_bytes == 0


# ---------------------------------------------------------------------------
# AudioScore
# ---------------------------------------------------------------------------


class TestAudioScore:
    def test_as_dict(self):
        s = AudioScore(metric_name="test", score=1.0, passed=True, details={"k": "v"})
        d = s.as_dict()
        assert d["metric_name"] == "test"
        assert d["score"] == 1.0
        assert d["passed"] is True
        assert d["details"] == {"k": "v"}

    def test_defaults(self):
        s = AudioScore(metric_name="test", score=0.0, passed=False)
        assert s.details == {}


# ---------------------------------------------------------------------------
# AudioEvalReport
# ---------------------------------------------------------------------------


class TestAudioEvalReport:
    def test_as_dict(self):
        scores = [AudioScore(metric_name="a", score=1.0, passed=True)]
        r = AudioEvalReport(scores=scores, overall_score=1.0, pass_rate=1.0)
        d = r.as_dict()
        assert len(d["scores"]) == 1
        assert d["overall_score"] == 1.0

    def test_from_dict(self):
        d = {
            "scores": [{"metric_name": "a", "score": 0.5, "passed": False, "details": {}}],
            "overall_score": 0.5,
            "pass_rate": 0.0,
        }
        r = AudioEvalReport.from_dict(d)
        assert len(r.scores) == 1
        assert r.scores[0].metric_name == "a"
        assert r.overall_score == 0.5

    def test_roundtrip(self):
        scores = [
            AudioScore(metric_name="a", score=1.0, passed=True),
            AudioScore(metric_name="b", score=0.0, passed=False),
        ]
        original = AudioEvalReport(scores=scores, overall_score=0.5, pass_rate=0.5)
        restored = AudioEvalReport.from_dict(original.as_dict())
        assert len(restored.scores) == 2
        assert restored.overall_score == original.overall_score

    def test_format_text(self):
        scores = [
            AudioScore(metric_name="exists", score=1.0, passed=True),
            AudioScore(metric_name="format", score=0.0, passed=False),
        ]
        r = AudioEvalReport(scores=scores, overall_score=0.5, pass_rate=0.5)
        text = r.format_text()
        assert "Audio Evaluation Report" in text
        assert "exists: 1.00 [PASS]" in text
        assert "format: 0.00 [FAIL]" in text
        assert "Overall score: 0.50" in text
        assert "Pass rate: 50.0%" in text

    def test_empty_report(self):
        r = AudioEvalReport()
        assert r.scores == []
        assert r.overall_score == 0.0


# ---------------------------------------------------------------------------
# check_audio_exists
# ---------------------------------------------------------------------------


class TestCheckAudioExists:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 100)
        result = check_audio_exists(str(f))
        assert result.passed is True
        assert result.score == 1.0

    def test_missing_file(self):
        result = check_audio_exists("/nonexistent/file.wav")
        assert result.passed is False
        assert result.details["reason"] == "file not found"

    def test_empty_path(self):
        result = check_audio_exists("")
        assert result.passed is False
        assert result.details["reason"] == "empty path"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.wav"
        f.write_bytes(b"")
        result = check_audio_exists(str(f))
        assert result.passed is False
        assert result.details["reason"] == "file is empty"


# ---------------------------------------------------------------------------
# check_audio_format
# ---------------------------------------------------------------------------


class TestCheckAudioFormat:
    def test_valid_wav(self):
        result = check_audio_format("song.wav")
        assert result.passed is True

    def test_valid_mp3(self):
        result = check_audio_format("song.mp3")
        assert result.passed is True

    def test_invalid_format(self):
        result = check_audio_format("song.txt")
        assert result.passed is False

    def test_custom_formats(self):
        result = check_audio_format("song.aac", expected_formats=["aac", "m4a"])
        assert result.passed is True

    def test_case_insensitive(self):
        result = check_audio_format("song.WAV")
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_audio_duration
# ---------------------------------------------------------------------------


class TestCheckAudioDuration:
    def test_within_bounds(self):
        meta = AudioMetadata(duration_seconds=10.0)
        result = check_audio_duration(meta)
        assert result.passed is True

    def test_too_short(self):
        meta = AudioMetadata(duration_seconds=0.01)
        result = check_audio_duration(meta)
        assert result.passed is False

    def test_too_long(self):
        meta = AudioMetadata(duration_seconds=600.0)
        result = check_audio_duration(meta)
        assert result.passed is False

    def test_custom_bounds(self):
        meta = AudioMetadata(duration_seconds=5.0)
        result = check_audio_duration(meta, min_seconds=1.0, max_seconds=10.0)
        assert result.passed is True

    def test_zero_duration(self):
        meta = AudioMetadata(duration_seconds=0.0)
        result = check_audio_duration(meta)
        assert result.passed is False

    def test_exact_min_boundary(self):
        meta = AudioMetadata(duration_seconds=0.1)
        result = check_audio_duration(meta, min_seconds=0.1)
        assert result.passed is True

    def test_exact_max_boundary(self):
        meta = AudioMetadata(duration_seconds=300.0)
        result = check_audio_duration(meta, max_seconds=300.0)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_audio_size
# ---------------------------------------------------------------------------


class TestCheckAudioSize:
    def test_within_bounds(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 500)
        result = check_audio_size(str(f))
        assert result.passed is True

    def test_too_small(self, tmp_path):
        f = tmp_path / "tiny.wav"
        f.write_bytes(b"\x00" * 10)
        result = check_audio_size(str(f))
        assert result.passed is False

    def test_missing_file(self):
        result = check_audio_size("/nonexistent/file.wav")
        assert result.passed is False
        assert result.details["reason"] == "file not found"

    def test_custom_bounds(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 200)
        result = check_audio_size(str(f), min_bytes=100, max_bytes=300)
        assert result.passed is True


# ---------------------------------------------------------------------------
# extract_audio_metadata_from_path
# ---------------------------------------------------------------------------


class TestExtractAudioMetadataFromPath:
    def test_with_existing_file(self, tmp_path):
        f = tmp_path / "song.mp3"
        f.write_bytes(b"\x00" * 256)
        meta = extract_audio_metadata_from_path(str(f))
        assert meta.format == "mp3"
        assert meta.size_bytes == 256

    def test_nonexistent_file(self):
        meta = extract_audio_metadata_from_path("/fake/song.flac")
        assert meta.format == "flac"
        assert meta.size_bytes == 0

    def test_no_extension(self, tmp_path):
        f = tmp_path / "noext"
        f.write_bytes(b"\x00" * 10)
        meta = extract_audio_metadata_from_path(str(f))
        assert meta.format == ""


# ---------------------------------------------------------------------------
# evaluate_audio
# ---------------------------------------------------------------------------


class TestEvaluateAudio:
    def test_full_pipeline_valid(self, tmp_path):
        f = tmp_path / "song.wav"
        f.write_bytes(b"\x00" * 500)
        meta = AudioMetadata(duration_seconds=5.0)
        report = evaluate_audio(str(f), metadata=meta)
        # exists, format, size, duration - all should pass
        assert report.pass_rate == 1.0
        assert report.overall_score == 1.0
        assert len(report.scores) == 4

    def test_missing_file(self):
        report = evaluate_audio("/nonexistent/song.wav")
        assert report.pass_rate < 1.0

    def test_no_metadata_skips_duration(self, tmp_path):
        f = tmp_path / "song.ogg"
        f.write_bytes(b"\x00" * 500)
        report = evaluate_audio(str(f))
        # duration check skipped since metadata.duration_seconds is 0
        metric_names = [s.metric_name for s in report.scores]
        assert "audio_duration" not in metric_names
        assert "audio_exists" in metric_names

    def test_with_duration_metadata(self, tmp_path):
        f = tmp_path / "song.flac"
        f.write_bytes(b"\x00" * 500)
        meta = AudioMetadata(duration_seconds=2.0)
        report = evaluate_audio(str(f), metadata=meta)
        metric_names = [s.metric_name for s in report.scores]
        assert "audio_duration" in metric_names
