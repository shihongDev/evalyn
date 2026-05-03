"""Argparse parser fixtures for ``test_introspect``.

Each factory returns a minimal parser exercising one ``ParamKind`` so the
classifier in ``evalyn_dashboard.introspect`` can be unit-tested in isolation,
independent of any real evalyn CLI command.
"""

from __future__ import annotations

import argparse


def make_bool_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bool-cmd", description="bool sample")
    p.add_argument("--flag", action="store_true", help="a boolean flag")
    return p


def make_select_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="select-cmd", description="select sample")
    p.add_argument("--mode", choices=["a", "b", "c"], default="a")
    return p


def make_multiselect_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="multi-cmd", description="multiselect sample")
    p.add_argument("--tags", nargs="*", choices=["x", "y", "z"], default=[])
    return p


def make_number_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="num-cmd", description="number sample")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    return p


def make_path_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="path-cmd", description="path sample")
    p.add_argument("--output", default="./out.json")
    p.add_argument("--input-file", required=True)
    return p


def make_long_text_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lt-cmd", description="longtext sample")
    p.add_argument("--prompt", default="")
    p.add_argument("--system-prompt", default="")
    return p


def make_string_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="str-cmd", description="string sample")
    p.add_argument("--name", required=True)
    return p
