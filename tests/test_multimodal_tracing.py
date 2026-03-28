from __future__ import annotations

from datetime import datetime, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.trace.multimodal import (
    MediaAttachment,
    MultimodalSpanData,
    MultimodalStats,
    compute_multimodal_stats,
    create_media_attachment,
    create_multimodal_span,
    extract_multimodal_data,
    has_media,
    supported_media_types,
)

import pytest


def _span(sid="s1", **attrs):
    return Span(
        id=sid,
        name="test",
        span_type="llm_call",
        parent_id=None,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# -- MediaAttachment --


class TestMediaAttachment:
    def test_as_dict(self):
        m = MediaAttachment(media_type="image", file_path="/img.png", mime_type="image/png", size_bytes=1024)
        d = m.as_dict()
        assert d["media_type"] == "image"
        assert d["file_path"] == "/img.png"
        assert d["size_bytes"] == 1024

    def test_from_dict(self):
        d = {"media_type": "audio", "file_path": "/a.wav", "mime_type": "audio/wav", "size_bytes": 2048}
        m = MediaAttachment.from_dict(d)
        assert m.media_type == "audio"
        assert m.file_path == "/a.wav"
        assert m.size_bytes == 2048

    def test_roundtrip(self):
        m = MediaAttachment(media_type="video", file_path="/v.mp4", mime_type="video/mp4", size_bytes=9999, metadata={"fps": 30})
        m2 = MediaAttachment.from_dict(m.as_dict())
        assert m2.media_type == m.media_type
        assert m2.file_path == m.file_path
        assert m2.metadata == m.metadata

    def test_defaults(self):
        m = MediaAttachment(media_type="document")
        assert m.file_path == ""
        assert m.mime_type == ""
        assert m.size_bytes == 0
        assert m.metadata == {}


# -- create_media_attachment --


class TestCreateMediaAttachment:
    def test_valid_image(self):
        m = create_media_attachment("image", file_path="/img.png")
        assert m.media_type == "image"

    def test_valid_audio(self):
        m = create_media_attachment("audio")
        assert m.media_type == "audio"

    def test_valid_video(self):
        m = create_media_attachment("video")
        assert m.media_type == "video"

    def test_valid_document(self):
        m = create_media_attachment("document")
        assert m.media_type == "document"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            create_media_attachment("spreadsheet")

    def test_passes_all_fields(self):
        m = create_media_attachment("image", file_path="/x.png", mime_type="image/png", size_bytes=512)
        assert m.file_path == "/x.png"
        assert m.mime_type == "image/png"
        assert m.size_bytes == 512


# -- MultimodalSpanData --


class TestMultimodalSpanData:
    def test_as_dict_empty(self):
        d = MultimodalSpanData().as_dict()
        assert d["input_media"] == []
        assert d["output_media"] == []
        assert d["text_input"] == ""

    def test_as_dict_with_media(self):
        m = MediaAttachment(media_type="image")
        data = MultimodalSpanData(input_media=[m], text_input="hello")
        d = data.as_dict()
        assert len(d["input_media"]) == 1
        assert d["text_input"] == "hello"

    def test_from_dict(self):
        raw = {
            "input_media": [{"media_type": "image"}],
            "output_media": [],
            "text_input": "q",
            "text_output": "a",
        }
        data = MultimodalSpanData.from_dict(raw)
        assert len(data.input_media) == 1
        assert data.text_input == "q"
        assert data.text_output == "a"

    def test_roundtrip(self):
        m1 = MediaAttachment(media_type="image", size_bytes=100)
        m2 = MediaAttachment(media_type="audio", size_bytes=200)
        data = MultimodalSpanData(input_media=[m1], output_media=[m2], text_input="in", text_output="out")
        data2 = MultimodalSpanData.from_dict(data.as_dict())
        assert len(data2.input_media) == 1
        assert len(data2.output_media) == 1
        assert data2.text_input == "in"
        assert data2.text_output == "out"


# -- MultimodalStats --


class TestMultimodalStats:
    def test_as_dict(self):
        s = MultimodalStats(total_media=3, by_type={"image": 2, "audio": 1}, total_size_bytes=500)
        d = s.as_dict()
        assert d["total_media"] == 3
        assert d["by_type"]["image"] == 2

    def test_format_text(self):
        s = MultimodalStats(total_media=1, by_type={"image": 1}, total_size_bytes=100)
        txt = s.format_text()
        assert "Total media: 1" in txt
        assert "Total size bytes: 100" in txt

    def test_defaults(self):
        s = MultimodalStats()
        assert s.total_media == 0
        assert s.by_type == {}
        assert s.total_size_bytes == 0


# -- create_multimodal_span / extract --


class TestCreateAndExtract:
    def test_create_span_basic(self):
        span = create_multimodal_span("s1", text_input="hello")
        assert span.id == "s1"
        assert "multimodal_data" in span.attributes

    def test_create_span_with_media(self):
        m = create_media_attachment("image", file_path="/i.png", size_bytes=100)
        span = create_multimodal_span("s2", input_media=[m])
        data = extract_multimodal_data(span)
        assert data is not None
        assert len(data.input_media) == 1
        assert data.input_media[0].file_path == "/i.png"

    def test_extract_roundtrip(self):
        m_in = create_media_attachment("audio", size_bytes=50)
        m_out = create_media_attachment("video", size_bytes=200)
        span = create_multimodal_span("s3", text_input="q", text_output="a", input_media=[m_in], output_media=[m_out])
        data = extract_multimodal_data(span)
        assert data is not None
        assert data.text_input == "q"
        assert data.text_output == "a"
        assert len(data.input_media) == 1
        assert len(data.output_media) == 1

    def test_extract_returns_none_no_data(self):
        span = _span()
        assert extract_multimodal_data(span) is None

    def test_extract_returns_none_bad_type(self):
        span = _span(multimodal_data="not_a_dict")
        assert extract_multimodal_data(span) is None


# -- has_media --


class TestHasMedia:
    def test_true_with_input_media(self):
        m = create_media_attachment("image")
        span = create_multimodal_span("s1", input_media=[m])
        assert has_media(span) is True

    def test_true_with_output_media(self):
        m = create_media_attachment("document")
        span = create_multimodal_span("s1", output_media=[m])
        assert has_media(span) is True

    def test_false_no_media(self):
        span = create_multimodal_span("s1", text_input="just text")
        assert has_media(span) is False

    def test_false_no_multimodal_data(self):
        span = _span()
        assert has_media(span) is False


# -- compute_multimodal_stats --


class TestComputeStats:
    def test_aggregation(self):
        m1 = create_media_attachment("image", size_bytes=100)
        m2 = create_media_attachment("image", size_bytes=200)
        m3 = create_media_attachment("audio", size_bytes=300)
        s1 = create_multimodal_span("s1", input_media=[m1, m2])
        s2 = create_multimodal_span("s2", output_media=[m3])
        stats = compute_multimodal_stats([s1, s2])
        assert stats.total_media == 3
        assert stats.by_type["image"] == 2
        assert stats.by_type["audio"] == 1
        assert stats.total_size_bytes == 600

    def test_empty_spans(self):
        stats = compute_multimodal_stats([])
        assert stats.total_media == 0
        assert stats.by_type == {}
        assert stats.total_size_bytes == 0

    def test_spans_without_multimodal_data(self):
        span = _span()
        stats = compute_multimodal_stats([span])
        assert stats.total_media == 0


# -- supported_media_types --


class TestSupportedMediaTypes:
    def test_returns_four(self):
        types = supported_media_types()
        assert len(types) == 4

    def test_contains_all(self):
        types = supported_media_types()
        assert "image" in types
        assert "audio" in types
        assert "video" in types
        assert "document" in types
