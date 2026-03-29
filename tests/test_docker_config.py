"""Tests for docker_config module."""

from __future__ import annotations

from evalyn_sdk.docker_config import (
    DockerConfig,
    format_docker_instructions,
    generate_docker_compose,
    generate_dockerfile,
    generate_dockerignore,
    validate_docker_config,
)


# ---------------------------------------------------------------------------
# DockerConfig
# ---------------------------------------------------------------------------


class TestDockerConfig:
    def test_defaults(self):
        cfg = DockerConfig()
        assert cfg.base_image == "python:3.11-slim"
        assert cfg.python_version == "3.11"
        assert cfg.include_dev_deps is False
        assert cfg.port == 0

    def test_custom_values(self):
        cfg = DockerConfig(
            base_image="python:3.12-slim",
            python_version="3.12",
            include_dev_deps=True,
            port=8080,
        )
        assert cfg.base_image == "python:3.12-slim"
        assert cfg.port == 8080

    def test_as_dict(self):
        cfg = DockerConfig(port=5000)
        d = cfg.as_dict()
        assert d["port"] == 5000
        assert d["base_image"] == "python:3.11-slim"

    def test_from_dict(self):
        d = {"base_image": "python:3.10", "python_version": "3.10", "port": 3000}
        cfg = DockerConfig.from_dict(d)
        assert cfg.base_image == "python:3.10"
        assert cfg.port == 3000

    def test_from_dict_defaults(self):
        cfg = DockerConfig.from_dict({})
        assert cfg.base_image == "python:3.11-slim"
        assert cfg.include_dev_deps is False

    def test_roundtrip(self):
        cfg = DockerConfig(base_image="python:3.12", python_version="3.12", port=9000)
        restored = DockerConfig.from_dict(cfg.as_dict())
        assert restored.as_dict() == cfg.as_dict()


# ---------------------------------------------------------------------------
# generate_dockerfile
# ---------------------------------------------------------------------------


class TestGenerateDockerfile:
    def test_default_config(self):
        df = generate_dockerfile(DockerConfig())
        assert "FROM python:3.11-slim" in df
        assert "WORKDIR /app" in df
        assert "ENTRYPOINT" in df

    def test_custom_base_image(self):
        cfg = DockerConfig(base_image="python:3.12-bookworm")
        df = generate_dockerfile(cfg)
        assert "FROM python:3.12-bookworm" in df

    def test_production_deps(self):
        cfg = DockerConfig(include_dev_deps=False)
        df = generate_dockerfile(cfg)
        assert "pip install --no-cache-dir -e '.'" in df
        assert ".[dev]" not in df

    def test_dev_deps(self):
        cfg = DockerConfig(include_dev_deps=True)
        df = generate_dockerfile(cfg)
        assert ".[dev]" in df

    def test_port_exposed(self):
        cfg = DockerConfig(port=8080)
        df = generate_dockerfile(cfg)
        assert "EXPOSE 8080" in df

    def test_no_port_when_zero(self):
        cfg = DockerConfig(port=0)
        df = generate_dockerfile(cfg)
        assert "EXPOSE" not in df

    def test_non_root_user(self):
        df = generate_dockerfile(DockerConfig())
        assert "useradd" in df
        assert "USER evalyn" in df

    def test_layer_caching_order(self):
        df = generate_dockerfile(DockerConfig())
        # pyproject.toml should be copied before full COPY .
        pyproject_pos = df.index("COPY pyproject.toml")
        copy_all_pos = df.index("COPY . .")
        assert pyproject_pos < copy_all_pos

    def test_entrypoint(self):
        df = generate_dockerfile(DockerConfig())
        assert "evalyn_sdk.cli.main" in df


# ---------------------------------------------------------------------------
# generate_dockerignore
# ---------------------------------------------------------------------------


class TestGenerateDockerignore:
    def test_contains_pycache(self):
        content = generate_dockerignore()
        assert "__pycache__/" in content

    def test_contains_venv(self):
        content = generate_dockerignore()
        assert ".venv/" in content

    def test_contains_git(self):
        content = generate_dockerignore()
        assert ".git/" in content

    def test_contains_pytest_cache(self):
        content = generate_dockerignore()
        assert ".pytest_cache/" in content

    def test_contains_docker_files(self):
        content = generate_dockerignore()
        assert "Dockerfile" in content
        assert ".dockerignore" in content


# ---------------------------------------------------------------------------
# generate_docker_compose
# ---------------------------------------------------------------------------


class TestGenerateDockerCompose:
    def test_basic_structure(self):
        compose = generate_docker_compose(DockerConfig())
        assert "version:" in compose
        assert "services:" in compose
        assert "evalyn:" in compose

    def test_volume_defined(self):
        compose = generate_docker_compose(DockerConfig())
        assert "volumes:" in compose
        assert "evalyn-data:" in compose

    def test_port_mapping(self):
        cfg = DockerConfig(port=8080)
        compose = generate_docker_compose(cfg)
        assert "8080:8080" in compose

    def test_no_port_when_zero(self):
        cfg = DockerConfig(port=0)
        compose = generate_docker_compose(cfg)
        assert "ports:" not in compose

    def test_dev_service(self):
        cfg = DockerConfig(include_dev_deps=True)
        compose = generate_docker_compose(cfg)
        assert "evalyn-dev:" in compose

    def test_no_dev_service_by_default(self):
        cfg = DockerConfig(include_dev_deps=False)
        compose = generate_docker_compose(cfg)
        assert "evalyn-dev:" not in compose

    def test_environment_set(self):
        compose = generate_docker_compose(DockerConfig())
        assert "EVALYN_ENV=production" in compose


# ---------------------------------------------------------------------------
# validate_docker_config
# ---------------------------------------------------------------------------


class TestValidateDockerConfig:
    def test_valid_config(self):
        errors = validate_docker_config(DockerConfig())
        assert errors == []

    def test_empty_base_image(self):
        cfg = DockerConfig(base_image="")
        errors = validate_docker_config(cfg)
        assert any("base_image" in e for e in errors)

    def test_empty_python_version(self):
        cfg = DockerConfig(python_version="")
        errors = validate_docker_config(cfg)
        assert any("python_version" in e for e in errors)

    def test_invalid_python_version_format(self):
        cfg = DockerConfig(python_version="3")
        errors = validate_docker_config(cfg)
        assert any("major.minor" in e for e in errors)

    def test_old_python_version(self):
        cfg = DockerConfig(python_version="3.7")
        errors = validate_docker_config(cfg)
        assert any("3.8" in e for e in errors)

    def test_python_2_rejected(self):
        cfg = DockerConfig(python_version="2.7")
        errors = validate_docker_config(cfg)
        assert any("3.8" in e for e in errors)

    def test_negative_port(self):
        cfg = DockerConfig(port=-1)
        errors = validate_docker_config(cfg)
        assert any("port" in e for e in errors)

    def test_port_too_high(self):
        cfg = DockerConfig(port=70000)
        errors = validate_docker_config(cfg)
        assert any("port" in e for e in errors)

    def test_valid_high_port(self):
        cfg = DockerConfig(port=65535)
        errors = validate_docker_config(cfg)
        assert errors == []

    def test_non_numeric_python_version(self):
        cfg = DockerConfig(python_version="abc.def")
        errors = validate_docker_config(cfg)
        assert any("numeric" in e for e in errors)


# ---------------------------------------------------------------------------
# format_docker_instructions
# ---------------------------------------------------------------------------


class TestFormatDockerInstructions:
    def test_contains_build_command(self):
        text = format_docker_instructions(DockerConfig())
        assert "docker build" in text

    def test_contains_run_command(self):
        text = format_docker_instructions(DockerConfig())
        assert "docker run" in text

    def test_contains_compose_command(self):
        text = format_docker_instructions(DockerConfig())
        assert "docker-compose up" in text

    def test_port_in_run_command(self):
        cfg = DockerConfig(port=8080)
        text = format_docker_instructions(cfg)
        assert "8080:8080" in text

    def test_no_port_flag_when_zero(self):
        cfg = DockerConfig(port=0)
        text = format_docker_instructions(cfg)
        assert "-p" not in text

    def test_shows_base_image(self):
        cfg = DockerConfig(base_image="python:3.12-slim")
        text = format_docker_instructions(cfg)
        assert "python:3.12-slim" in text

    def test_shows_dev_status(self):
        text_no = format_docker_instructions(DockerConfig(include_dev_deps=False))
        text_yes = format_docker_instructions(DockerConfig(include_dev_deps=True))
        assert "no" in text_no
        assert "yes" in text_yes
