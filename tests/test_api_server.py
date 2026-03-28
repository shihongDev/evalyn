"""Tests for evalyn_sdk.integration.api_server.

Run with: uv run pytest tests/test_api_server.py -v
"""
from __future__ import annotations

import pytest

from evalyn_sdk.integration.api_server import (
    APIConfig,
    APIRequest,
    APIResponse,
    APIRoute,
    APIRouter,
    create_default_router,
    health_handler,
)


# ---------------------------------------------------------------------------
# APIRoute
# ---------------------------------------------------------------------------


class TestAPIRoute:
    def test_as_dict(self):
        route = APIRoute(method="GET", path="/health", handler_name="health_handler")
        d = route.as_dict()
        assert d["method"] == "GET"
        assert d["path"] == "/health"
        assert d["handler_name"] == "health_handler"
        assert d["description"] == ""

    def test_as_dict_with_description(self):
        route = APIRoute(
            method="POST", path="/eval", handler_name="eval_handler", description="Run eval"
        )
        assert route.as_dict()["description"] == "Run eval"


# ---------------------------------------------------------------------------
# APIRequest
# ---------------------------------------------------------------------------


class TestAPIRequest:
    def test_as_dict(self):
        req = APIRequest(method="GET", path="/runs")
        d = req.as_dict()
        assert d["method"] == "GET"
        assert d["path"] == "/runs"
        assert d["query_params"] == {}
        assert d["body"] is None
        assert d["headers"] == {}

    def test_from_dict_defaults(self):
        req = APIRequest.from_dict({})
        assert req.method == "GET"
        assert req.path == "/"

    def test_from_dict_full(self):
        data = {
            "method": "POST",
            "path": "/evaluate",
            "query_params": {"verbose": "true"},
            "body": {"dataset": "test"},
            "headers": {"Authorization": "Bearer tok"},
        }
        req = APIRequest.from_dict(data)
        assert req.method == "POST"
        assert req.path == "/evaluate"
        assert req.query_params == {"verbose": "true"}
        assert req.body == {"dataset": "test"}
        assert req.headers["Authorization"] == "Bearer tok"

    def test_roundtrip(self):
        req = APIRequest(
            method="PUT", path="/runs/1", body={"name": "updated"}, headers={"X-Key": "val"}
        )
        restored = APIRequest.from_dict(req.as_dict())
        assert restored.method == req.method
        assert restored.path == req.path
        assert restored.body == req.body
        assert restored.headers == req.headers


# ---------------------------------------------------------------------------
# APIResponse
# ---------------------------------------------------------------------------


class TestAPIResponse:
    def test_defaults(self):
        resp = APIResponse()
        assert resp.status_code == 200
        assert resp.is_success is True

    def test_is_success_true(self):
        for code in (200, 201, 202, 301, 399):
            assert APIResponse(status_code=code).is_success is True

    def test_is_success_false(self):
        for code in (400, 404, 500):
            assert APIResponse(status_code=code).is_success is False

    def test_as_dict(self):
        resp = APIResponse(status_code=404, body={"error": "nope"}, error="nope")
        d = resp.as_dict()
        assert d["status_code"] == 404
        assert d["error"] == "nope"

    def test_roundtrip(self):
        resp = APIResponse(status_code=201, body={"id": "abc"}, headers={"X-Req": "123"})
        d = resp.as_dict()
        # Manual reconstruction (APIResponse has no from_dict)
        resp2 = APIResponse(
            status_code=d["status_code"],
            body=d["body"],
            headers=d["headers"],
            error=d["error"],
        )
        assert resp2.status_code == resp.status_code
        assert resp2.body == resp.body


# ---------------------------------------------------------------------------
# APIConfig
# ---------------------------------------------------------------------------


class TestAPIConfig:
    def test_defaults(self):
        cfg = APIConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080
        assert cfg.debug is False
        assert cfg.cors_origins == ["*"]
        assert cfg.api_key_required is False

    def test_from_dict(self):
        cfg = APIConfig.from_dict({"port": 9090, "debug": True})
        assert cfg.port == 9090
        assert cfg.debug is True

    def test_roundtrip(self):
        cfg = APIConfig(host="127.0.0.1", port=3000, cors_origins=["http://localhost"])
        restored = APIConfig.from_dict(cfg.as_dict())
        assert restored.host == cfg.host
        assert restored.port == cfg.port
        assert restored.cors_origins == cfg.cors_origins

    def test_no_hardcoded_api_key(self):
        cfg = APIConfig()
        d = cfg.as_dict()
        # Ensure no api_key value is stored in config
        assert "api_key" not in d


# ---------------------------------------------------------------------------
# APIRouter
# ---------------------------------------------------------------------------


class TestAPIRouter:
    def test_register_and_list(self):
        router = APIRouter()
        router.register("GET", "/foo", health_handler, description="Foo")
        routes = router.list_routes()
        assert len(routes) == 1
        assert routes[0].path == "/foo"
        assert routes[0].method == "GET"

    def test_match_exact(self):
        router = APIRouter()
        router.register("GET", "/health", health_handler)
        route = router.match("GET", "/health")
        assert route is not None
        assert route.handler_name == "health_handler"

    def test_match_case_insensitive_method(self):
        router = APIRouter()
        router.register("get", "/health", health_handler)
        route = router.match("GET", "/health")
        assert route is not None

    def test_match_path_params(self):
        router = APIRouter()
        router.register("GET", "/runs/{id}", health_handler)
        route = router.match("GET", "/runs/abc123")
        assert route is not None

    def test_match_no_match(self):
        router = APIRouter()
        router.register("GET", "/health", health_handler)
        assert router.match("POST", "/health") is None
        assert router.match("GET", "/missing") is None

    def test_handle_success(self):
        router = APIRouter()
        router.register("GET", "/health", health_handler)
        req = APIRequest(method="GET", path="/health")
        resp = router.handle(req)
        assert resp.status_code == 200
        assert resp.body["status"] == "ok"

    def test_handle_404(self):
        router = APIRouter()
        req = APIRequest(method="GET", path="/nonexistent")
        resp = router.handle(req)
        assert resp.status_code == 404
        assert resp.is_success is False

    def test_empty_router(self):
        router = APIRouter()
        assert router.list_routes() == []
        assert router.match("GET", "/anything") is None

    def test_duplicate_routes(self):
        router = APIRouter()
        router.register("GET", "/a", health_handler)
        router.register("GET", "/a", health_handler)
        # Both registrations kept
        assert len(router.list_routes()) == 2

    def test_generate_openapi_spec(self):
        router = APIRouter()
        router.register("GET", "/health", health_handler, description="Health check")
        router.register("POST", "/evaluate", health_handler, description="Run eval")
        spec = router.generate_openapi_spec()
        assert spec["openapi"] == "3.0.0"
        assert "info" in spec
        assert spec["info"]["title"] == "Evalyn API"
        assert "/health" in spec["paths"]
        assert "get" in spec["paths"]["/health"]
        assert "/evaluate" in spec["paths"]
        assert "post" in spec["paths"]["/evaluate"]

    def test_generate_openapi_spec_empty(self):
        router = APIRouter()
        spec = router.generate_openapi_spec()
        assert spec["paths"] == {}


# ---------------------------------------------------------------------------
# create_default_router
# ---------------------------------------------------------------------------


class TestDefaultRouter:
    def test_has_routes(self):
        router = create_default_router()
        routes = router.list_routes()
        assert len(routes) >= 5

    def test_health_route(self):
        router = create_default_router()
        route = router.match("GET", "/health")
        assert route is not None

    def test_runs_route(self):
        router = create_default_router()
        assert router.match("GET", "/runs") is not None

    def test_run_detail_route(self):
        router = create_default_router()
        assert router.match("GET", "/runs/some-id") is not None

    def test_evaluate_route(self):
        router = create_default_router()
        assert router.match("POST", "/evaluate") is not None

    def test_metrics_route(self):
        router = create_default_router()
        assert router.match("GET", "/metrics") is not None


# ---------------------------------------------------------------------------
# health_handler
# ---------------------------------------------------------------------------


class TestHealthHandler:
    def test_returns_ok(self):
        req = APIRequest(method="GET", path="/health")
        resp = health_handler(req)
        assert resp.status_code == 200
        assert resp.body == {"status": "ok"}
        assert resp.is_success is True
