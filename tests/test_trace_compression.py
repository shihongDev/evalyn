from __future__ import annotations

import base64
import gzip

from datetime import datetime, timedelta, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.trace.compression import (
    CompressionConfig,
    CompressionStats,
    compress_span_attributes,
    compress_text,
    compute_compression_stats,
    decompress_span_attributes,
    decompress_text,
)


def _span(sid, name="s", span_type="llm_call", parent_id=None, start_offset_ms=0, duration_ms=100, **attrs):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = base + timedelta(milliseconds=start_offset_ms)
    return Span(id=sid, name=name, span_type=span_type, parent_id=parent_id,
                start_time=t, end_time=t + timedelta(milliseconds=duration_ms), attributes=attrs)


# -- compress_text / decompress_text roundtrip --


class TestCompressDecompress:
    def test_roundtrip_basic(self):
        text = "Hello, world! " * 100
        compressed = compress_text(text)
        assert decompress_text(compressed) == text

    def test_roundtrip_unicode(self):
        text = "Unicode test: cafe, nino, resume " * 50
        assert decompress_text(compress_text(text)) == text

    def test_roundtrip_empty(self):
        assert decompress_text(compress_text("")) == ""

    def test_compressed_is_bytes(self):
        result = compress_text("some text")
        assert isinstance(result, bytes)

    def test_compressed_smaller_for_large_input(self):
        text = "a" * 10000
        compressed = compress_text(text)
        assert len(compressed) < len(text.encode("utf-8"))

    def test_level_parameter(self):
        text = "x" * 5000
        low = compress_text(text, level=1)
        high = compress_text(text, level=9)
        # Both should decompress to the same thing
        assert decompress_text(low) == text
        assert decompress_text(high) == text


# -- CompressionConfig --


class TestCompressionConfig:
    def test_defaults(self):
        cfg = CompressionConfig()
        assert cfg.min_size_bytes == 1000
        assert cfg.level == 6
        assert cfg.enabled is True

    def test_as_dict(self):
        cfg = CompressionConfig(min_size_bytes=500, level=3, enabled=False)
        d = cfg.as_dict()
        assert d["min_size_bytes"] == 500
        assert d["level"] == 3
        assert d["enabled"] is False

    def test_from_dict_roundtrip(self):
        original = CompressionConfig(min_size_bytes=2000, level=9, enabled=False)
        restored = CompressionConfig.from_dict(original.as_dict())
        assert restored.min_size_bytes == original.min_size_bytes
        assert restored.level == original.level
        assert restored.enabled == original.enabled

    def test_from_dict_defaults(self):
        cfg = CompressionConfig.from_dict({})
        assert cfg.min_size_bytes == 1000
        assert cfg.level == 6
        assert cfg.enabled is True


# -- CompressionStats --


class TestCompressionStats:
    def test_savings_pct(self):
        stats = CompressionStats(original_bytes=1000, compressed_bytes=300, compression_ratio=0.3, fields_compressed=1)
        assert abs(stats.savings_pct - 70.0) < 0.01

    def test_savings_pct_zero_original(self):
        stats = CompressionStats(original_bytes=0, compressed_bytes=0, compression_ratio=1.0, fields_compressed=0)
        assert stats.savings_pct == 0.0

    def test_savings_pct_no_savings(self):
        stats = CompressionStats(original_bytes=100, compressed_bytes=100, compression_ratio=1.0, fields_compressed=1)
        assert abs(stats.savings_pct) < 0.01

    def test_as_dict_contains_savings(self):
        stats = CompressionStats(original_bytes=1000, compressed_bytes=400, compression_ratio=0.4, fields_compressed=2)
        d = stats.as_dict()
        assert d["savings_pct"] == stats.savings_pct
        assert d["fields_compressed"] == 2


# -- compress_span_attributes --


class TestCompressSpanAttributes:
    def test_creates_new_span(self):
        s = _span("s1", input="x" * 2000)
        cfg = CompressionConfig()
        new_s, _ = compress_span_attributes(s, cfg)
        assert new_s is not s
        assert new_s.id == s.id

    def test_original_unchanged(self):
        original_input = "y" * 2000
        s = _span("s1", input=original_input)
        cfg = CompressionConfig()
        compress_span_attributes(s, cfg)
        assert s.attributes["input"] == original_input

    def test_large_string_compressed(self):
        s = _span("s1", input="z" * 2000)
        cfg = CompressionConfig(min_size_bytes=1000)
        new_s, stats = compress_span_attributes(s, cfg)
        attr = new_s.attributes["input"]
        assert isinstance(attr, dict)
        assert attr["__compressed__"] is True
        assert "data" in attr
        assert "original_size" in attr
        assert stats.fields_compressed == 1

    def test_small_string_not_compressed(self):
        s = _span("s1", input="short")
        cfg = CompressionConfig(min_size_bytes=1000)
        new_s, stats = compress_span_attributes(s, cfg)
        assert new_s.attributes["input"] == "short"
        assert stats.fields_compressed == 0

    def test_non_string_attribute_preserved(self):
        s = _span("s1", count=42, flag=True)
        cfg = CompressionConfig()
        new_s, stats = compress_span_attributes(s, cfg)
        assert new_s.attributes["count"] == 42
        assert new_s.attributes["flag"] is True
        assert stats.fields_compressed == 0

    def test_disabled_config(self):
        s = _span("s1", input="a" * 5000)
        cfg = CompressionConfig(enabled=False)
        new_s, stats = compress_span_attributes(s, cfg)
        assert new_s.attributes["input"] == "a" * 5000
        assert stats.fields_compressed == 0

    def test_base64_marker_present(self):
        s = _span("s1", input="w" * 3000)
        cfg = CompressionConfig()
        new_s, _ = compress_span_attributes(s, cfg)
        attr = new_s.attributes["input"]
        # Verify the data field is valid base64
        decoded = base64.b64decode(attr["data"])
        assert isinstance(decoded, bytes)
        # And it decompresses to the original
        assert gzip.decompress(decoded).decode("utf-8") == "w" * 3000

    def test_multiple_fields_compressed(self):
        s = _span("s1", input="a" * 2000, output="b" * 2000)
        cfg = CompressionConfig()
        new_s, stats = compress_span_attributes(s, cfg)
        assert stats.fields_compressed == 2
        assert isinstance(new_s.attributes["input"], dict)
        assert isinstance(new_s.attributes["output"], dict)


# -- decompress_span_attributes --


class TestDecompressSpanAttributes:
    def test_roundtrip(self):
        s = _span("s1", input="hello world " * 200, output="result " * 200)
        cfg = CompressionConfig(min_size_bytes=100)
        compressed_s, _ = compress_span_attributes(s, cfg)
        decompressed_s = decompress_span_attributes(compressed_s)
        assert decompressed_s.attributes["input"] == s.attributes["input"]
        assert decompressed_s.attributes["output"] == s.attributes["output"]

    def test_creates_new_span(self):
        s = _span("s1", input="data " * 500)
        cfg = CompressionConfig(min_size_bytes=100)
        compressed_s, _ = compress_span_attributes(s, cfg)
        decompressed_s = decompress_span_attributes(compressed_s)
        assert decompressed_s is not compressed_s

    def test_no_compressed_markers_passthrough(self):
        s = _span("s1", input="plain text")
        result = decompress_span_attributes(s)
        assert result.attributes["input"] == "plain text"

    def test_non_dict_attributes_untouched(self):
        s = _span("s1", count=99)
        result = decompress_span_attributes(s)
        assert result.attributes["count"] == 99


# -- compute_compression_stats --


class TestComputeCompressionStats:
    def test_estimates_savings(self):
        spans = [
            _span("s1", input="a" * 3000),
            _span("s2", input="b" * 3000),
        ]
        cfg = CompressionConfig()
        stats = compute_compression_stats(spans, cfg)
        assert stats.original_bytes > 0
        assert stats.compressed_bytes > 0
        assert stats.compressed_bytes < stats.original_bytes
        assert stats.fields_compressed == 2

    def test_empty_list(self):
        stats = compute_compression_stats([], CompressionConfig())
        assert stats.original_bytes == 0
        assert stats.fields_compressed == 0

    def test_nothing_to_compress(self):
        spans = [_span("s1", input="tiny")]
        cfg = CompressionConfig(min_size_bytes=1000)
        stats = compute_compression_stats(spans, cfg)
        assert stats.fields_compressed == 0
        assert stats.original_bytes == 0
