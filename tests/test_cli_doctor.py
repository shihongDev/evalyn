"""Tests for `evalyn doctor` CLI command.

Covers each check function in isolation (no subprocess) plus a few end-to-end
sanity checks that confirm `run_all_checks` produces a coherent list.

Run with: pytest tests/test_cli_doctor.py -v
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

# Make sure the SDK is importable (conftest.py also does this, but tests in
# this file should not rely on test collection order).
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.cli.commands import doctor
from evalyn_sdk.cli.commands.doctor import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    _parse_requires_python,
    check_api_keys,
    check_config_file,
    check_evalyn_version,
    check_optional_deps,
    check_project_shape,
    check_python_version,
    check_required_deps,
    check_storage,
    render_summary,
    render_table,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# _parse_requires_python
# ---------------------------------------------------------------------------


class TestParseRequiresPython:
    def test_simple_version(self):
        assert _parse_requires_python(">=3.10") == (3, 10)

    def test_with_whitespace(self):
        assert _parse_requires_python("  >=3.11  ") == (3, 11)

    def test_major_only(self):
        assert _parse_requires_python(">=3") == (3, 0)

    def test_three_segment(self):
        # We only consume major.minor; patch is ignored.
        assert _parse_requires_python(">=3.12.0") == (3, 12)

    def test_no_operator_returns_none(self):
        assert _parse_requires_python("3.10") is None

    def test_empty_returns_none(self):
        assert _parse_requires_python("") is None
        assert _parse_requires_python(None) is None

    def test_garbage_returns_none(self):
        assert _parse_requires_python(">=abc") is None


# ---------------------------------------------------------------------------
# check_python_version
# ---------------------------------------------------------------------------


class TestPythonVersionCheck:
    def test_running_python_passes(self):
        # Tests run on the SDK's supported interpreter so this must PASS.
        result = check_python_version()
        assert result.status == PASS
        assert "." in result.detail  # contains a version string

    def test_pyproject_lookup_falls_back_when_missing(self, tmp_path, monkeypatch):
        # Pointing the lookup at an empty temp dir should still produce the
        # documented default of (3, 10) and pass on this interpreter.
        result_default = doctor._read_min_python_from_pyproject(start=tmp_path)
        assert result_default == (3, 10)


# ---------------------------------------------------------------------------
# check_evalyn_version
# ---------------------------------------------------------------------------


class TestEvalynVersionCheck:
    def test_version_string_is_returned(self):
        import evalyn_sdk

        result = check_evalyn_version()
        assert result.status == PASS
        assert result.detail == evalyn_sdk.__version__


# ---------------------------------------------------------------------------
# check_required_deps
# ---------------------------------------------------------------------------


class TestRequiredDeps:
    def test_all_required_present(self):
        # The SDK declares opentelemetry/yaml as required (or stdlib for sqlite);
        # in a dev environment all three should be importable.
        results = check_required_deps()
        names = {r.name for r in results}
        assert any("opentelemetry" in n for n in names)
        assert any("sqlite3" in n for n in names)
        assert any("PyYAML" in n for n in names)
        for r in results:
            assert r.status == PASS, f"{r.name} not installed in test env: {r.detail}"


# ---------------------------------------------------------------------------
# check_optional_deps
# ---------------------------------------------------------------------------


class TestOptionalDeps:
    def test_missing_optional_is_warn_not_fail(self, monkeypatch):
        # Force every optional import to fail; the check must produce WARN
        # rows, never FAIL, so a fresh user environment still doctors clean.
        def always_fail(name):
            raise ImportError(name)

        monkeypatch.setattr(doctor.importlib, "import_module", always_fail)
        results = check_optional_deps()
        # One result per optional dep.
        assert len(results) == len(doctor.OPTIONAL_DEPS)
        for r in results:
            assert r.status == WARN
            assert "not installed" in r.detail

    def test_present_optional_is_pass(self, monkeypatch):
        # Pretend every import succeeds. The test does not care about version
        # extraction here, just the PASS status.
        class _FakeMod:
            __version__ = "9.9.9"

        monkeypatch.setattr(doctor.importlib, "import_module", lambda name: _FakeMod())
        results = check_optional_deps()
        for r in results:
            assert r.status == PASS


# ---------------------------------------------------------------------------
# check_api_keys
# ---------------------------------------------------------------------------


class TestApiKeys:
    def test_all_keys_missing(self):
        results = check_api_keys(env={})
        assert len(results) == len(doctor.API_KEY_VARS)
        for r in results:
            assert r.status == WARN
            assert r.detail == "(not set)"

    def test_all_keys_set(self):
        env = {var: "secret-do-not-log" for var in doctor.API_KEY_VARS}
        results = check_api_keys(env=env)
        for r in results:
            assert r.status == PASS
            assert r.detail == "(set)"

    def test_value_is_never_in_detail(self):
        secret = "sk-test-NEVER-LEAK"  # pragma: allowlist secret
        env = {"OPENAI_API_KEY": secret}
        results = check_api_keys(env=env)
        for r in results:
            assert secret not in r.detail
            assert secret not in r.name

    def test_partial_set(self):
        env = {"OPENAI_API_KEY": "x"}
        results = check_api_keys(env=env)
        by_name = {r.name: r for r in results}
        assert by_name["API key: OPENAI_API_KEY"].status == PASS
        assert by_name["API key: ANTHROPIC_API_KEY"].status == WARN


# ---------------------------------------------------------------------------
# check_config_file
# ---------------------------------------------------------------------------


class TestConfigFile:
    def test_no_config_is_skip(self, tmp_path):
        result = check_config_file(search_root=tmp_path)
        assert result.status == SKIP

    def test_valid_yaml_is_pass(self, tmp_path):
        cfg = tmp_path / "evalyn.yaml"
        cfg.write_text("project: demo\nllm:\n  model: gpt-4o-mini\n")
        result = check_config_file(search_root=tmp_path)
        assert result.status == PASS
        assert "evalyn.yaml" in result.detail

    def test_invalid_yaml_is_fail(self, tmp_path):
        cfg = tmp_path / "evalyn.yaml"
        # Unclosed bracket - definitely a parse error.
        cfg.write_text("project: [unclosed\n  - broken\n")
        result = check_config_file(search_root=tmp_path)
        assert result.status == FAIL

    def test_evalynrc_is_found(self, tmp_path):
        cfg = tmp_path / ".evalynrc"
        cfg.write_text("project: demo\n")
        result = check_config_file(search_root=tmp_path)
        assert result.status == PASS
        assert ".evalynrc" in result.detail


# ---------------------------------------------------------------------------
# check_storage
# ---------------------------------------------------------------------------


class TestStorage:
    # conftest.py sets EVALYN_DB globally so all tests share one trace DB.
    # check_storage honors that env var, which would mask the "no DB" path.
    # Unset it here so each test sees a clean filesystem-only view.
    @pytest.fixture(autouse=True)
    def _clear_evalyn_db(self, monkeypatch):
        monkeypatch.delenv("EVALYN_DB", raising=False)

    def test_writable_no_db_is_skip(self, tmp_path):
        results = check_storage(search_root=tmp_path)
        statuses = {r.name: r for r in results}
        assert statuses["Storage writable"].status == PASS
        assert statuses["Trace DB"].status == SKIP

    def test_existing_evalyn_dir_is_used(self, tmp_path):
        (tmp_path / ".evalyn").mkdir()
        results = check_storage(search_root=tmp_path)
        statuses = {r.name: r for r in results}
        assert statuses["Storage writable"].status == PASS
        assert ".evalyn/" in statuses["Storage writable"].detail

    def test_readable_db_is_pass(self, tmp_path):
        traces = tmp_path / ".evalyn" / "traces"
        traces.mkdir(parents=True)
        db = traces / "prod.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
        conn.close()
        results = check_storage(search_root=tmp_path)
        statuses = {r.name: r for r in results}
        assert statuses["Trace DB"].status == PASS
        assert "prod.db" in statuses["Trace DB"].detail

    def test_unwritable_target_is_fail(self, tmp_path, monkeypatch):
        # Patch NamedTemporaryFile so the writability probe raises.
        def boom(*_args, **_kwargs):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(doctor.tempfile, "NamedTemporaryFile", boom)
        results = check_storage(search_root=tmp_path)
        statuses = {r.name: r for r in results}
        assert statuses["Storage writable"].status == FAIL


# ---------------------------------------------------------------------------
# check_project_shape
# ---------------------------------------------------------------------------


class TestProjectShape:
    def test_empty_dir_is_warn(self, tmp_path):
        result = check_project_shape(search_root=tmp_path)
        assert result.status == WARN

    def test_evalyn_yaml_is_pass(self, tmp_path):
        (tmp_path / "evalyn.yaml").write_text("project: demo\n")
        result = check_project_shape(search_root=tmp_path)
        assert result.status == PASS
        assert "evalyn.yaml" in result.detail

    def test_dot_evalyn_is_pass(self, tmp_path):
        (tmp_path / ".evalyn").mkdir()
        result = check_project_shape(search_root=tmp_path)
        assert result.status == PASS
        assert ".evalyn/" in result.detail

    def test_dataset_is_pass(self, tmp_path):
        dset = tmp_path / "data" / "myproj"
        dset.mkdir(parents=True)
        (dset / "dataset.jsonl").write_text('{"id": 1}\n')
        result = check_project_shape(search_root=tmp_path)
        assert result.status == PASS
        assert "dataset.jsonl" in result.detail


# ---------------------------------------------------------------------------
# run_all_checks + rendering
# ---------------------------------------------------------------------------


class TestRunAll:
    def test_runs_without_exception(self):
        results = run_all_checks()
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, CheckResult)
            assert r.status in (PASS, FAIL, WARN, SKIP)

    def test_includes_every_section(self):
        results = run_all_checks()
        names = [r.name for r in results]
        assert "Python version" in names
        assert "evalyn version" in names
        assert any(n.startswith("Required:") for n in names)
        assert any(n.startswith("Optional:") for n in names)
        assert any(n.startswith("API key:") for n in names)
        assert "Config file" in names
        assert "Storage writable" in names
        assert "Project shape" in names


class TestRendering:
    def test_render_table_contains_all_rows(self):
        results = [
            CheckResult("A", PASS, "ok"),
            CheckResult("B", FAIL, "boom"),
            CheckResult("C", WARN, "meh"),
            CheckResult("D", SKIP, "n/a"),
        ]
        out = render_table(results)
        for name in ("A", "B", "C", "D"):
            assert name in out
        for status in (PASS, FAIL, WARN, SKIP):
            assert status in out

    def test_render_summary_counts(self):
        results = [
            CheckResult("a", PASS),
            CheckResult("b", PASS),
            CheckResult("c", FAIL),
            CheckResult("d", WARN),
            CheckResult("e", SKIP),
        ]
        summary = render_summary(results)
        assert "PASS: 2" in summary
        assert "FAIL: 1" in summary
        assert "WARN: 1" in summary
        assert "SKIP: 1" in summary


# ---------------------------------------------------------------------------
# cmd_doctor exit code
# ---------------------------------------------------------------------------


class TestCmdDoctorExit:
    def test_exits_nonzero_when_any_fail(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "run_all_checks",
            lambda: [CheckResult("synthetic", FAIL, "rigged")],
        )
        import argparse

        with pytest.raises(SystemExit) as excinfo:
            doctor.cmd_doctor(argparse.Namespace())
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "synthetic" in out
        assert "FAIL" in out

    def test_exits_zero_when_all_pass(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "run_all_checks",
            lambda: [CheckResult("synthetic", PASS, "fine")],
        )
        import argparse

        # No SystemExit means clean exit (returncode 0 from CLI standpoint).
        doctor.cmd_doctor(argparse.Namespace())
        out = capsys.readouterr().out
        assert "PASS" in out
