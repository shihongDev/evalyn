from .base import StorageBackend


def __getattr__(name: str):
    if name == "SQLiteStorage":
        from .sqlite import SQLiteStorage

        globals()["SQLiteStorage"] = SQLiteStorage
        return SQLiteStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["StorageBackend", "SQLiteStorage"]
