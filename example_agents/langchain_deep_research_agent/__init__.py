import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graph import graph

__all__ = ["graph"]
