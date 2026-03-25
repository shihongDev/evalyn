"""
Tests for evalyn_sdk.utils.api_client retry logic.

Run with: pytest tests/test_api_client.py -v
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock
import urllib.error
import urllib.request

import pytest

from evalyn_sdk.utils.api_client import _http_post


def _make_http_error(code: int, body: str = "error") -> urllib.error.HTTPError:
    """Create an HTTPError with a readable body."""
    import io
    err = urllib.error.HTTPError(
        url="https://example.com",
        code=code,
        msg=f"HTTP {code}",
        hdrs={},
        fp=io.BytesIO(body.encode("utf-8")),
    )
    return err


def _make_url_error(reason: str = "Connection refused") -> urllib.error.URLError:
    return urllib.error.URLError(reason)


class TestHttpPostRetry:
    """Tests for _http_post retry with exponential backoff."""

    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_success_no_retry(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps({"ok": True}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = _http_post("https://api.test/v1", {}, {}, 30, "Test")
        assert result == {"ok": True}
        assert mock_urlopen.call_count == 1

    @patch("evalyn_sdk.utils.api_client.time.sleep")
    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_retry_on_429(self, mock_urlopen, mock_sleep):
        # First call: 429, second call: success
        resp_ok = MagicMock()
        resp_ok.read.return_value = json.dumps({"ok": True}).encode()
        resp_ok.__enter__ = lambda s: s
        resp_ok.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            _make_http_error(429, "rate limited"),
            resp_ok,
        ]

        result = _http_post("https://api.test/v1", {}, {}, 30, "Test")
        assert result == {"ok": True}
        assert mock_urlopen.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("evalyn_sdk.utils.api_client.time.sleep")
    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_retry_on_503(self, mock_urlopen, mock_sleep):
        resp_ok = MagicMock()
        resp_ok.read.return_value = json.dumps({"ok": True}).encode()
        resp_ok.__enter__ = lambda s: s
        resp_ok.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            _make_http_error(503, "service unavailable"),
            resp_ok,
        ]

        result = _http_post("https://api.test/v1", {}, {}, 30, "Test")
        assert result == {"ok": True}
        assert mock_urlopen.call_count == 2

    @patch("evalyn_sdk.utils.api_client.time.sleep")
    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_retry_on_500(self, mock_urlopen, mock_sleep):
        resp_ok = MagicMock()
        resp_ok.read.return_value = json.dumps({"ok": True}).encode()
        resp_ok.__enter__ = lambda s: s
        resp_ok.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            _make_http_error(500),
            _make_http_error(502),
            resp_ok,
        ]

        result = _http_post("https://api.test/v1", {}, {}, 30, "Test")
        assert result == {"ok": True}
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_no_retry_on_400(self, mock_urlopen):
        """Client errors (except 429) raise immediately."""
        mock_urlopen.side_effect = _make_http_error(400, "bad request")

        with pytest.raises(RuntimeError, match="400"):
            _http_post("https://api.test/v1", {}, {}, 30, "Test")
        assert mock_urlopen.call_count == 1

    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_no_retry_on_401(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(401, "unauthorized")

        with pytest.raises(RuntimeError, match="401"):
            _http_post("https://api.test/v1", {}, {}, 30, "Test")
        assert mock_urlopen.call_count == 1

    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_no_retry_on_404(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(404, "not found")

        with pytest.raises(RuntimeError, match="404"):
            _http_post("https://api.test/v1", {}, {}, 30, "Test")
        assert mock_urlopen.call_count == 1

    @patch("evalyn_sdk.utils.api_client.time.sleep")
    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_exhausted_retries_raises(self, mock_urlopen, mock_sleep):
        """After max retries, raises RuntimeError."""
        mock_urlopen.side_effect = _make_http_error(429, "rate limited")

        with pytest.raises(RuntimeError, match="429"):
            _http_post("https://api.test/v1", {}, {}, 30, "Test")
        # 1 initial + 3 retries = 4 total
        assert mock_urlopen.call_count == 4

    @patch("evalyn_sdk.utils.api_client.time.sleep")
    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_retry_on_url_error(self, mock_urlopen, mock_sleep):
        """Connection errors are retried."""
        resp_ok = MagicMock()
        resp_ok.read.return_value = json.dumps({"ok": True}).encode()
        resp_ok.__enter__ = lambda s: s
        resp_ok.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            _make_url_error("Connection refused"),
            resp_ok,
        ]

        result = _http_post("https://api.test/v1", {}, {}, 30, "Test")
        assert result == {"ok": True}
        assert mock_urlopen.call_count == 2

    @patch("evalyn_sdk.utils.api_client.time.sleep")
    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_url_error_exhausted_retries(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = _make_url_error("Connection refused")

        with pytest.raises(RuntimeError, match="connection error"):
            _http_post("https://api.test/v1", {}, {}, 30, "Test")
        assert mock_urlopen.call_count == 4

    @patch("evalyn_sdk.utils.api_client.time.sleep")
    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_backoff_delays_increase(self, mock_urlopen, mock_sleep):
        """Verify delays follow exponential pattern."""
        mock_urlopen.side_effect = _make_http_error(429, "rate limited")

        with pytest.raises(RuntimeError):
            _http_post("https://api.test/v1", {}, {}, 30, "Test")

        # Check sleep was called with increasing delays
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert len(delays) == 3
        # Base delays: 1, 2, 4 (plus up to 0.5 jitter)
        assert 1.0 <= delays[0] <= 1.5
        assert 2.0 <= delays[1] <= 2.5
        assert 4.0 <= delays[2] <= 4.5

    @patch("evalyn_sdk.utils.api_client.time.sleep")
    @patch("evalyn_sdk.utils.api_client.urllib.request.urlopen")
    def test_retry_on_504(self, mock_urlopen, mock_sleep):
        resp_ok = MagicMock()
        resp_ok.read.return_value = json.dumps({"ok": True}).encode()
        resp_ok.__enter__ = lambda s: s
        resp_ok.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            _make_http_error(504, "gateway timeout"),
            resp_ok,
        ]

        result = _http_post("https://api.test/v1", {}, {}, 30, "Test")
        assert result == {"ok": True}
