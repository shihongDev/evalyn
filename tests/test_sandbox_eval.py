"""Tests for the sandboxed agent evaluation module."""

from __future__ import annotations

from evalyn_sdk.sandbox_eval import (
    SandboxConfig,
    SandboxResult,
    SandboxSpec,
    build_docker_command,
    check_docker_available,
    create_sandbox_spec,
    estimate_sandbox_cost,
    format_sandbox_plan,
    validate_sandbox_config,
)


# ---------------------------------------------------------------------------
# SandboxConfig
# ---------------------------------------------------------------------------


class TestSandboxConfig:
    def test_defaults(self):
        cfg = SandboxConfig()
        assert cfg.timeout_seconds == 30
        assert cfg.max_memory_mb == 256
        assert cfg.network_enabled is False
        assert cfg.image == "python:3.11-slim"
        assert cfg.working_dir == "/workspace"

    def test_custom_values(self):
        cfg = SandboxConfig(
            timeout_seconds=60,
            max_memory_mb=512,
            network_enabled=True,
            image="node:18",
            working_dir="/app",
        )
        assert cfg.timeout_seconds == 60
        assert cfg.max_memory_mb == 512
        assert cfg.network_enabled is True
        assert cfg.image == "node:18"
        assert cfg.working_dir == "/app"

    def test_as_dict(self):
        cfg = SandboxConfig()
        d = cfg.as_dict()
        assert d["timeout_seconds"] == 30
        assert d["max_memory_mb"] == 256
        assert d["network_enabled"] is False
        assert d["image"] == "python:3.11-slim"
        assert d["working_dir"] == "/workspace"

    def test_from_dict_full(self):
        d = {
            "timeout_seconds": 120,
            "max_memory_mb": 1024,
            "network_enabled": True,
            "image": "ruby:3.2",
            "working_dir": "/home",
        }
        cfg = SandboxConfig.from_dict(d)
        assert cfg.timeout_seconds == 120
        assert cfg.max_memory_mb == 1024
        assert cfg.network_enabled is True
        assert cfg.image == "ruby:3.2"
        assert cfg.working_dir == "/home"

    def test_from_dict_defaults(self):
        cfg = SandboxConfig.from_dict({})
        assert cfg.timeout_seconds == 30
        assert cfg.max_memory_mb == 256

    def test_roundtrip(self):
        cfg = SandboxConfig(timeout_seconds=45, max_memory_mb=128, network_enabled=True)
        restored = SandboxConfig.from_dict(cfg.as_dict())
        assert restored.timeout_seconds == cfg.timeout_seconds
        assert restored.max_memory_mb == cfg.max_memory_mb
        assert restored.network_enabled == cfg.network_enabled
        assert restored.image == cfg.image
        assert restored.working_dir == cfg.working_dir


# ---------------------------------------------------------------------------
# SandboxResult
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_basic(self):
        r = SandboxResult(exit_code=0, stdout="hello", stderr="")
        assert r.exit_code == 0
        assert r.stdout == "hello"
        assert r.stderr == ""
        assert r.timed_out is False
        assert r.duration_seconds == 0.0

    def test_timed_out(self):
        r = SandboxResult(
            exit_code=124, stdout="", stderr="timeout", timed_out=True, duration_seconds=30.0
        )
        assert r.timed_out is True
        assert r.duration_seconds == 30.0

    def test_as_dict(self):
        r = SandboxResult(exit_code=1, stdout="out", stderr="err", duration_seconds=2.5)
        d = r.as_dict()
        assert d["exit_code"] == 1
        assert d["stdout"] == "out"
        assert d["stderr"] == "err"
        assert d["timed_out"] is False
        assert d["duration_seconds"] == 2.5

    def test_from_dict_full(self):
        d = {
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
            "duration_seconds": 1.2,
        }
        r = SandboxResult.from_dict(d)
        assert r.exit_code == 0
        assert r.stdout == "ok"
        assert r.duration_seconds == 1.2

    def test_from_dict_minimal(self):
        r = SandboxResult.from_dict({"exit_code": 0})
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.timed_out is False

    def test_roundtrip(self):
        r = SandboxResult(
            exit_code=2, stdout="data", stderr="warn", timed_out=True, duration_seconds=5.5
        )
        restored = SandboxResult.from_dict(r.as_dict())
        assert restored.exit_code == r.exit_code
        assert restored.stdout == r.stdout
        assert restored.stderr == r.stderr
        assert restored.timed_out == r.timed_out
        assert restored.duration_seconds == r.duration_seconds


# ---------------------------------------------------------------------------
# SandboxSpec
# ---------------------------------------------------------------------------


class TestSandboxSpec:
    def test_basic(self):
        cfg = SandboxConfig()
        spec = SandboxSpec(sandbox_id="abc-123", config=cfg)
        assert spec.sandbox_id == "abc-123"
        assert spec.status == "ready"

    def test_custom_status(self):
        spec = SandboxSpec(sandbox_id="x", config=SandboxConfig(), status="running")
        assert spec.status == "running"

    def test_as_dict(self):
        spec = SandboxSpec(sandbox_id="s1", config=SandboxConfig(), status="completed")
        d = spec.as_dict()
        assert d["sandbox_id"] == "s1"
        assert d["status"] == "completed"
        assert "timeout_seconds" in d["config"]

    def test_from_dict(self):
        d = {
            "sandbox_id": "s2",
            "config": {"timeout_seconds": 10, "max_memory_mb": 64},
            "status": "failed",
        }
        spec = SandboxSpec.from_dict(d)
        assert spec.sandbox_id == "s2"
        assert spec.config.timeout_seconds == 10
        assert spec.status == "failed"

    def test_from_dict_defaults(self):
        spec = SandboxSpec.from_dict({"sandbox_id": "s3"})
        assert spec.status == "ready"
        assert spec.config.timeout_seconds == 30

    def test_roundtrip(self):
        cfg = SandboxConfig(timeout_seconds=90, network_enabled=True)
        spec = SandboxSpec(sandbox_id="rt", config=cfg, status="running")
        restored = SandboxSpec.from_dict(spec.as_dict())
        assert restored.sandbox_id == spec.sandbox_id
        assert restored.status == spec.status
        assert restored.config.timeout_seconds == 90
        assert restored.config.network_enabled is True


# ---------------------------------------------------------------------------
# build_docker_command
# ---------------------------------------------------------------------------


class TestBuildDockerCommand:
    def test_basic_command(self):
        cfg = SandboxConfig()
        cmd = build_docker_command("print('hi')", cfg)
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--rm" in cmd
        assert "print('hi')" in cmd

    def test_memory_flag(self):
        cfg = SandboxConfig(max_memory_mb=512)
        cmd = build_docker_command("x=1", cfg)
        idx = cmd.index("--memory")
        assert cmd[idx + 1] == "512m"

    def test_network_disabled(self):
        cfg = SandboxConfig(network_enabled=False)
        cmd = build_docker_command("x=1", cfg)
        idx = cmd.index("--network")
        assert cmd[idx + 1] == "none"

    def test_network_enabled(self):
        cfg = SandboxConfig(network_enabled=True)
        cmd = build_docker_command("x=1", cfg)
        assert "--network" not in cmd

    def test_working_dir(self):
        cfg = SandboxConfig(working_dir="/data")
        cmd = build_docker_command("x=1", cfg)
        idx = cmd.index("--workdir")
        assert cmd[idx + 1] == "/data"

    def test_timeout_in_command(self):
        cfg = SandboxConfig(timeout_seconds=45)
        cmd = build_docker_command("x=1", cfg)
        idx = cmd.index("timeout")
        assert cmd[idx + 1] == "45"

    def test_image_in_command(self):
        cfg = SandboxConfig(image="python:3.12")
        cmd = build_docker_command("x=1", cfg)
        assert "python:3.12" in cmd

    def test_code_preserved(self):
        code = "import os; print(os.getcwd())"
        cmd = build_docker_command(code, SandboxConfig())
        assert code in cmd

    def test_returns_list_of_strings(self):
        cmd = build_docker_command("x=1", SandboxConfig())
        assert isinstance(cmd, list)
        assert all(isinstance(s, str) for s in cmd)


# ---------------------------------------------------------------------------
# validate_sandbox_config
# ---------------------------------------------------------------------------


class TestValidateSandboxConfig:
    def test_valid_default(self):
        errors = validate_sandbox_config(SandboxConfig())
        assert errors == []

    def test_zero_timeout(self):
        cfg = SandboxConfig(timeout_seconds=0)
        errors = validate_sandbox_config(cfg)
        assert any("timeout" in e for e in errors)

    def test_negative_timeout(self):
        cfg = SandboxConfig(timeout_seconds=-5)
        errors = validate_sandbox_config(cfg)
        assert any("timeout" in e for e in errors)

    def test_excessive_timeout(self):
        cfg = SandboxConfig(timeout_seconds=7200)
        errors = validate_sandbox_config(cfg)
        assert any("timeout" in e for e in errors)

    def test_memory_too_low(self):
        cfg = SandboxConfig(max_memory_mb=8)
        errors = validate_sandbox_config(cfg)
        assert any("memory" in e.lower() for e in errors)

    def test_memory_too_high(self):
        cfg = SandboxConfig(max_memory_mb=100000)
        errors = validate_sandbox_config(cfg)
        assert any("memory" in e.lower() for e in errors)

    def test_empty_image(self):
        cfg = SandboxConfig(image="")
        errors = validate_sandbox_config(cfg)
        assert any("image" in e for e in errors)

    def test_empty_working_dir(self):
        cfg = SandboxConfig(working_dir="")
        errors = validate_sandbox_config(cfg)
        assert any("working_dir" in e for e in errors)

    def test_multiple_errors(self):
        cfg = SandboxConfig(timeout_seconds=-1, max_memory_mb=1, image="", working_dir="")
        errors = validate_sandbox_config(cfg)
        assert len(errors) >= 3

    def test_boundary_timeout_valid(self):
        cfg = SandboxConfig(timeout_seconds=3600)
        errors = validate_sandbox_config(cfg)
        assert errors == []

    def test_boundary_memory_valid(self):
        cfg = SandboxConfig(max_memory_mb=16)
        errors = validate_sandbox_config(cfg)
        assert errors == []


# ---------------------------------------------------------------------------
# estimate_sandbox_cost
# ---------------------------------------------------------------------------


class TestEstimateSandboxCost:
    def test_basic_estimate(self):
        cfg = SandboxConfig(timeout_seconds=60, max_memory_mb=256)
        result = estimate_sandbox_cost(10, 5.0, cfg)
        assert result["n_items"] == 10
        assert result["avg_duration"] == 5.0
        assert result["total_seconds"] == 50.0
        assert result["wall_clock_seconds"] == 50.0

    def test_memory_seconds(self):
        cfg = SandboxConfig(max_memory_mb=512)
        result = estimate_sandbox_cost(4, 10.0, cfg)
        assert result["total_memory_mb_seconds"] == 4 * 10.0 * 512

    def test_capped_duration(self):
        cfg = SandboxConfig(timeout_seconds=5)
        result = estimate_sandbox_cost(3, 10.0, cfg)
        assert result["capped_total_seconds"] == 15.0  # 3 * min(10, 5) = 15

    def test_zero_items(self):
        result = estimate_sandbox_cost(0, 5.0, SandboxConfig())
        assert result["total_seconds"] == 0.0

    def test_keys_present(self):
        result = estimate_sandbox_cost(1, 1.0, SandboxConfig())
        expected_keys = {
            "n_items",
            "avg_duration",
            "total_seconds",
            "capped_total_seconds",
            "wall_clock_seconds",
            "total_memory_mb_seconds",
            "timeout_seconds",
            "max_memory_mb",
        }
        assert expected_keys == set(result.keys())


# ---------------------------------------------------------------------------
# create_sandbox_spec
# ---------------------------------------------------------------------------


class TestCreateSandboxSpec:
    def test_creates_with_uuid(self):
        spec = create_sandbox_spec()
        assert len(spec.sandbox_id) == 36  # UUID format
        assert spec.status == "ready"

    def test_default_config(self):
        spec = create_sandbox_spec()
        assert spec.config.timeout_seconds == 30

    def test_custom_config(self):
        cfg = SandboxConfig(timeout_seconds=120, image="golang:1.21")
        spec = create_sandbox_spec(cfg)
        assert spec.config.timeout_seconds == 120
        assert spec.config.image == "golang:1.21"

    def test_unique_ids(self):
        specs = [create_sandbox_spec() for _ in range(10)]
        ids = {s.sandbox_id for s in specs}
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# format_sandbox_plan
# ---------------------------------------------------------------------------


class TestFormatSandboxPlan:
    def test_empty_list(self):
        result = format_sandbox_plan([])
        assert "No sandboxes" in result

    def test_single_spec(self):
        spec = create_sandbox_spec()
        result = format_sandbox_plan([spec])
        assert "1 sandbox" in result
        assert spec.sandbox_id in result
        assert "ready" in result

    def test_multiple_specs(self):
        specs = [create_sandbox_spec() for _ in range(3)]
        result = format_sandbox_plan(specs)
        assert "3 sandbox" in result
        for spec in specs:
            assert spec.sandbox_id in result

    def test_contains_config_info(self):
        cfg = SandboxConfig(timeout_seconds=90, max_memory_mb=1024, network_enabled=True)
        spec = create_sandbox_spec(cfg)
        result = format_sandbox_plan([spec])
        assert "90" in result
        assert "1024" in result
        assert "enabled" in result

    def test_network_disabled_shown(self):
        cfg = SandboxConfig(network_enabled=False)
        spec = create_sandbox_spec(cfg)
        result = format_sandbox_plan([spec])
        assert "disabled" in result


# ---------------------------------------------------------------------------
# check_docker_available
# ---------------------------------------------------------------------------


class TestCheckDockerAvailable:
    def test_returns_bool(self):
        result = check_docker_available()
        assert isinstance(result, bool)
