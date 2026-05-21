"""Metric namespacing: organize metrics by project or team namespace."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NamespacedMetric:
    """A metric scoped to a namespace."""

    namespace: str
    metric_id: str
    full_id: str  # "namespace/metric_id"
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "metric_id": self.metric_id,
            "full_id": self.full_id,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NamespacedMetric:
        return cls(
            namespace=data["namespace"],
            metric_id=data["metric_id"],
            full_id=data["full_id"],
            description=data.get("description", ""),
        )


@dataclass
class Namespace:
    """A named collection of metric IDs."""

    name: str
    description: str = ""
    metrics: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "metrics": list(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Namespace:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            metrics=list(data.get("metrics", [])),
        )


def parse_full_id(full_id: str) -> tuple[str, str]:
    """Split 'namespace/metric' into (namespace, metric).

    If no '/' is present, the namespace defaults to 'default'.
    """
    if "/" in full_id:
        parts = full_id.split("/", 1)
        return (parts[0], parts[1])
    return ("default", full_id)


class NamespaceRegistry:
    """Registry for organizing metrics under namespaces."""

    def __init__(self) -> None:
        self._namespaces: dict[str, Namespace] = {}
        self._metrics: dict[str, NamespacedMetric] = {}  # keyed by full_id

    def register_namespace(self, name: str, description: str = "") -> None:
        """Register a new namespace (or update its description)."""
        if name in self._namespaces:
            self._namespaces[name].description = description
        else:
            self._namespaces[name] = Namespace(name=name, description=description)

    def add_metric(self, namespace: str, metric_id: str, description: str = "") -> NamespacedMetric:
        """Register a metric under a namespace. Creates the namespace if needed."""
        if namespace not in self._namespaces:
            self.register_namespace(namespace)
        full_id = f"{namespace}/{metric_id}"
        metric = NamespacedMetric(
            namespace=namespace,
            metric_id=metric_id,
            full_id=full_id,
            description=description,
        )
        self._metrics[full_id] = metric
        ns = self._namespaces[namespace]
        if metric_id not in ns.metrics:
            ns.metrics.append(metric_id)
        return metric

    def resolve(self, full_id: str) -> NamespacedMetric | None:
        """Look up a metric by its full 'namespace/metric' ID."""
        ns_name, metric_id = parse_full_id(full_id)
        key = f"{ns_name}/{metric_id}"
        return self._metrics.get(key)

    def search(self, query: str, namespace: str | None = None) -> list[NamespacedMetric]:
        """Search metrics by metric_id substring, optionally filtered by namespace."""
        results: list[NamespacedMetric] = []
        for m in self._metrics.values():
            if namespace is not None and m.namespace != namespace:
                continue
            if query in m.metric_id:
                results.append(m)
        return results

    def list_namespaces(self) -> list[Namespace]:
        """Return all registered namespaces."""
        return list(self._namespaces.values())

    def list_metrics(self, namespace: str | None = None) -> list[NamespacedMetric]:
        """List all metrics, optionally filtered by namespace."""
        if namespace is not None:
            return [m for m in self._metrics.values() if m.namespace == namespace]
        return list(self._metrics.values())

    def compare_across_namespaces(self, metric_id: str) -> list[NamespacedMetric]:
        """Find the same metric_id across different namespaces."""
        return [m for m in self._metrics.values() if m.metric_id == metric_id]


def format_namespace_tree(registry: NamespaceRegistry) -> str:
    """Render a tree-style display of namespaces and their metrics."""
    lines: list[str] = []
    namespaces = registry.list_namespaces()
    for ns in namespaces:
        desc = f" - {ns.description}" if ns.description else ""
        lines.append(f"{ns.name}/{desc}")
        metrics = registry.list_metrics(ns.name)
        for j, m in enumerate(metrics):
            is_last_m = j == len(metrics) - 1
            connector = "  `-- " if is_last_m else "  |-- "
            m_desc = f" ({m.description})" if m.description else ""
            lines.append(f"{connector}{m.metric_id}{m_desc}")
    return "\n".join(lines)
