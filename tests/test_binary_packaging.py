"""Tests for binary_packaging module."""

from __future__ import annotations

from evalyn_sdk.binary_packaging import (
    VALID_METHODS,
    PackagingConfig,
    estimate_binary_size,
    format_packaging_instructions,
    generate_build_script,
    generate_pyinstaller_spec,
    validate_packaging_config,
)


# ---------------------------------------------------------------------------
# PackagingConfig
# ---------------------------------------------------------------------------


class TestPackagingConfig:
    def test_defaults(self):
        cfg = PackagingConfig()
        assert cfg.method == "pyinstaller"
        assert cfg.entry_point == "evalyn_sdk.cli.main:main"
        assert cfg.output_name == "evalyn"
        assert cfg.one_file is True

    def test_custom_values(self):
        cfg = PackagingConfig(
            method="nuitka",
            entry_point="myapp:run",
            output_name="mybin",
            one_file=False,
        )
        assert cfg.method == "nuitka"
        assert cfg.output_name == "mybin"

    def test_as_dict(self):
        cfg = PackagingConfig(method="shiv", output_name="tool")
        d = cfg.as_dict()
        assert d["method"] == "shiv"
        assert d["output_name"] == "tool"

    def test_from_dict(self):
        d = {"method": "nuitka", "entry_point": "app:go", "output_name": "app", "one_file": False}
        cfg = PackagingConfig.from_dict(d)
        assert cfg.method == "nuitka"
        assert cfg.one_file is False

    def test_from_dict_defaults(self):
        cfg = PackagingConfig.from_dict({})
        assert cfg.method == "pyinstaller"
        assert cfg.one_file is True

    def test_roundtrip(self):
        cfg = PackagingConfig(method="nuitka", output_name="test", one_file=False)
        restored = PackagingConfig.from_dict(cfg.as_dict())
        assert restored.as_dict() == cfg.as_dict()


# ---------------------------------------------------------------------------
# generate_pyinstaller_spec
# ---------------------------------------------------------------------------


class TestGeneratePyinstallerSpec:
    def test_contains_analysis(self):
        spec = generate_pyinstaller_spec(PackagingConfig())
        assert "Analysis" in spec

    def test_contains_exe(self):
        spec = generate_pyinstaller_spec(PackagingConfig())
        assert "EXE" in spec

    def test_output_name(self):
        cfg = PackagingConfig(output_name="mytool")
        spec = generate_pyinstaller_spec(cfg)
        assert "mytool" in spec

    def test_entry_point_path(self):
        spec = generate_pyinstaller_spec(PackagingConfig())
        assert "evalyn_sdk/cli/main.py" in spec

    def test_one_file_mode(self):
        cfg = PackagingConfig(one_file=True)
        spec = generate_pyinstaller_spec(cfg)
        assert "exclude_binaries=False" in spec

    def test_one_dir_mode(self):
        cfg = PackagingConfig(one_file=False)
        spec = generate_pyinstaller_spec(cfg)
        assert "exclude_binaries=True" in spec

    def test_hidden_imports(self):
        spec = generate_pyinstaller_spec(PackagingConfig())
        assert "evalyn_sdk" in spec


# ---------------------------------------------------------------------------
# generate_build_script
# ---------------------------------------------------------------------------


class TestGenerateBuildScript:
    def test_pyinstaller_script(self):
        script = generate_build_script(PackagingConfig())
        assert "#!/usr/bin/env bash" in script
        assert "pyinstaller" in script
        assert "--onefile" in script

    def test_pyinstaller_onedir(self):
        cfg = PackagingConfig(one_file=False)
        script = generate_build_script(cfg)
        assert "--onedir" in script

    def test_nuitka_script(self):
        cfg = PackagingConfig(method="nuitka")
        script = generate_build_script(cfg)
        assert "nuitka" in script
        assert "--onefile" in script

    def test_nuitka_standalone(self):
        cfg = PackagingConfig(method="nuitka", one_file=False)
        script = generate_build_script(cfg)
        assert "--standalone" in script

    def test_shiv_script(self):
        cfg = PackagingConfig(method="shiv")
        script = generate_build_script(cfg)
        assert "shiv" in script
        assert ".pyz" in script

    def test_set_euo_pipefail(self):
        script = generate_build_script(PackagingConfig())
        assert "set -euo pipefail" in script

    def test_custom_output_name(self):
        cfg = PackagingConfig(output_name="custom-bin")
        script = generate_build_script(cfg)
        assert "custom-bin" in script


# ---------------------------------------------------------------------------
# estimate_binary_size
# ---------------------------------------------------------------------------


class TestEstimateBinarySize:
    def test_pyinstaller_onefile(self):
        result = estimate_binary_size(PackagingConfig())
        assert "30MB" in result
        assert "pyinstaller" in result

    def test_pyinstaller_onedir(self):
        cfg = PackagingConfig(one_file=False)
        result = estimate_binary_size(cfg)
        assert "50MB" in result

    def test_nuitka_onefile(self):
        cfg = PackagingConfig(method="nuitka")
        result = estimate_binary_size(cfg)
        assert "20MB" in result

    def test_nuitka_standalone(self):
        cfg = PackagingConfig(method="nuitka", one_file=False)
        result = estimate_binary_size(cfg)
        assert "35MB" in result

    def test_shiv(self):
        cfg = PackagingConfig(method="shiv")
        result = estimate_binary_size(cfg)
        assert "5MB" in result

    def test_unknown_method(self):
        cfg = PackagingConfig(method="unknown")
        result = estimate_binary_size(cfg)
        assert "unknown" in result


# ---------------------------------------------------------------------------
# validate_packaging_config
# ---------------------------------------------------------------------------


class TestValidatePackagingConfig:
    def test_valid_config(self):
        errors = validate_packaging_config(PackagingConfig())
        assert errors == []

    def test_invalid_method(self):
        cfg = PackagingConfig(method="invalid")
        errors = validate_packaging_config(cfg)
        assert any("method" in e for e in errors)

    def test_empty_entry_point(self):
        cfg = PackagingConfig(entry_point="")
        errors = validate_packaging_config(cfg)
        assert any("entry_point" in e for e in errors)

    def test_missing_colon_in_entry_point(self):
        cfg = PackagingConfig(entry_point="module.path")
        errors = validate_packaging_config(cfg)
        assert any("module.path:function" in e for e in errors)

    def test_empty_output_name(self):
        cfg = PackagingConfig(output_name="")
        errors = validate_packaging_config(cfg)
        assert any("output_name" in e for e in errors)

    def test_shiv_one_dir_rejected(self):
        cfg = PackagingConfig(method="shiv", one_file=False)
        errors = validate_packaging_config(cfg)
        assert any("shiv" in e for e in errors)

    def test_shiv_one_file_ok(self):
        cfg = PackagingConfig(method="shiv", one_file=True)
        errors = validate_packaging_config(cfg)
        assert errors == []

    def test_all_valid_methods(self):
        for method in VALID_METHODS:
            cfg = PackagingConfig(method=method)
            errors = validate_packaging_config(cfg)
            assert not any("method" in e for e in errors)


# ---------------------------------------------------------------------------
# format_packaging_instructions
# ---------------------------------------------------------------------------


class TestFormatPackagingInstructions:
    def test_pyinstaller_instructions(self):
        text = format_packaging_instructions(PackagingConfig())
        assert "pyinstaller" in text.lower()
        assert "pip install pyinstaller" in text

    def test_nuitka_instructions(self):
        cfg = PackagingConfig(method="nuitka")
        text = format_packaging_instructions(cfg)
        assert "nuitka" in text.lower()
        assert "C compiler" in text

    def test_shiv_instructions(self):
        cfg = PackagingConfig(method="shiv")
        text = format_packaging_instructions(cfg)
        assert "shiv" in text.lower()
        assert ".pyz" in text

    def test_shows_entry_point(self):
        text = format_packaging_instructions(PackagingConfig())
        assert "evalyn_sdk.cli.main:main" in text

    def test_shows_output_name(self):
        cfg = PackagingConfig(output_name="my-tool")
        text = format_packaging_instructions(cfg)
        assert "my-tool" in text

    def test_shows_estimated_size(self):
        text = format_packaging_instructions(PackagingConfig())
        assert "MB" in text

    def test_shows_output_location(self):
        text = format_packaging_instructions(PackagingConfig())
        assert "dist/" in text
