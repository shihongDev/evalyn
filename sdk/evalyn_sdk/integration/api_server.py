from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class APIRoute:
    """A single API route definition."""

    method: str  # "GET" / "POST" / "PUT" / "DELETE"
    path: str
    handler_name: str
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "handler_name": self.handler_name,
            "description": self.description,
        }


@dataclass
class APIRequest:
    """Incoming API request."""

    method: str
    path: str
    query_params: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "query_params": dict(self.query_params),
            "body": self.body,
            "headers": dict(self.headers),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APIRequest:
        return cls(
            method=data.get("method", "GET"),
            path=data.get("path", "/"),
            query_params=data.get("query_params", {}),
            body=data.get("body"),
            headers=data.get("headers", {}),
        )


@dataclass
class APIResponse:
    """Outgoing API response."""

    status_code: int = 200
    body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "body": dict(self.body),
            "headers": dict(self.headers),
            "error": self.error,
        }

    @property
    def is_success(self) -> bool:
        return self.status_code < 400


@dataclass
class APIConfig:
    """Configuration for the API server."""

    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    api_key_required: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "cors_origins": list(self.cors_origins),
            "api_key_required": self.api_key_required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APIConfig:
        return cls(
            host=data.get("host", "0.0.0.0"),
            port=data.get("port", 8080),
            debug=data.get("debug", False),
            cors_origins=data.get("cors_origins", ["*"]),
            api_key_required=data.get("api_key_required", False),
        )


class APIRouter:
    """Route requests to handlers."""

    def __init__(self) -> None:
        self._routes: list[APIRoute] = []
        self._handlers: dict[str, Callable] = {}

    def register(
        self,
        method: str,
        path: str,
        handler: Callable,
        description: str = "",
    ) -> None:
        """Register a route with its handler."""
        handler_name = handler.__name__
        route = APIRoute(
            method=method.upper(),
            path=path,
            handler_name=handler_name,
            description=description,
        )
        self._routes.append(route)
        self._handlers[f"{method.upper()} {path}"] = handler

    def match(self, method: str, path: str) -> APIRoute | None:
        """Find matching route for method and path.

        Supports simple path parameter matching: a route registered as
        /runs/{id} will match /runs/abc123.
        """
        method = method.upper()
        # Try exact match first
        for route in self._routes:
            if route.method == method and route.path == path:
                return route
        # Try pattern match (path parameters)
        for route in self._routes:
            if route.method != method:
                continue
            if _path_matches(route.path, path):
                return route
        return None

    def handle(self, request: APIRequest) -> APIResponse:
        """Route a request to its handler. Returns 404 if no match."""
        route = self.match(request.method, request.path)
        if route is None:
            return APIResponse(
                status_code=404,
                body={"error": "not found"},
                error="not found",
            )
        key = f"{route.method} {route.path}"
        handler = self._handlers[key]
        return handler(request)

    def list_routes(self) -> list[APIRoute]:
        """Return all registered routes."""
        return list(self._routes)

    def generate_openapi_spec(self) -> dict[str, Any]:
        """Generate a basic OpenAPI 3.0 spec from registered routes."""
        paths: dict[str, Any] = {}
        for route in self._routes:
            method_lower = route.method.lower()
            if route.path not in paths:
                paths[route.path] = {}
            paths[route.path][method_lower] = {
                "summary": route.description or route.handler_name,
                "responses": {
                    "200": {"description": "Success"},
                },
            }
        return {
            "openapi": "3.0.0",
            "info": {"title": "Evalyn API", "version": "1.0.0"},
            "paths": paths,
        }


def _path_matches(pattern: str, path: str) -> bool:
    """Check if a path matches a route pattern with {param} segments."""
    pattern_parts = pattern.strip("/").split("/")
    path_parts = path.strip("/").split("/")
    if len(pattern_parts) != len(path_parts):
        return False
    for pp, actual in zip(pattern_parts, path_parts):
        if pp.startswith("{") and pp.endswith("}"):
            continue
        if pp != actual:
            return False
    return True


# -- Built-in handlers -------------------------------------------------------


def health_handler(request: APIRequest) -> APIResponse:
    """Return health check response."""
    return APIResponse(status_code=200, body={"status": "ok"})


def _runs_handler(request: APIRequest) -> APIResponse:
    return APIResponse(status_code=200, body={"runs": []})


def _run_detail_handler(request: APIRequest) -> APIResponse:
    return APIResponse(status_code=200, body={"run": None})


def _evaluate_handler(request: APIRequest) -> APIResponse:
    return APIResponse(status_code=202, body={"status": "accepted"})


def _metrics_handler(request: APIRequest) -> APIResponse:
    return APIResponse(status_code=200, body={"metrics": []})


def create_default_router() -> APIRouter:
    """Create a router with built-in routes."""
    router = APIRouter()
    router.register("GET", "/health", health_handler, description="Health check")
    router.register("GET", "/runs", _runs_handler, description="List eval runs")
    router.register(
        "GET", "/runs/{id}", _run_detail_handler, description="Get eval run by id"
    )
    router.register("POST", "/evaluate", _evaluate_handler, description="Start evaluation")
    router.register("GET", "/metrics", _metrics_handler, description="List metrics")
    return router
