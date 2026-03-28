from __future__ import annotations

from evalyn_sdk.trace.selective_instrumentation import (
    InstrumentationConfig,
    InstrumentationTarget,
    create_exclude_config,
    create_include_config,
    list_instrumented_targets,
    match_target,
    should_instrument,
)


# ---------------------------------------------------------------------------
# InstrumentationTarget
# ---------------------------------------------------------------------------


class TestInstrumentationTarget:
    def test_full_path_all_parts(self):
        t = InstrumentationTarget(module_name="mymod", class_name="MyClass", method_name="run")
        assert t.full_path == "mymod.MyClass.run"

    def test_full_path_module_only(self):
        t = InstrumentationTarget(module_name="mymod")
        assert t.full_path == "mymod"

    def test_full_path_module_and_class(self):
        t = InstrumentationTarget(module_name="mymod", class_name="Foo")
        assert t.full_path == "mymod.Foo"

    def test_as_dict(self):
        t = InstrumentationTarget(module_name="m", class_name="C", method_name="f", enabled=False)
        d = t.as_dict()
        assert d == {"module_name": "m", "class_name": "C", "method_name": "f", "enabled": False}

    def test_from_dict(self):
        d = {"module_name": "m", "class_name": "C", "method_name": "f", "enabled": True}
        t = InstrumentationTarget.from_dict(d)
        assert t.full_path == "m.C.f"
        assert t.enabled is True

    def test_from_dict_defaults(self):
        t = InstrumentationTarget.from_dict({"module_name": "x"})
        assert t.class_name == ""
        assert t.method_name == ""
        assert t.enabled is True


# ---------------------------------------------------------------------------
# InstrumentationConfig
# ---------------------------------------------------------------------------


class TestInstrumentationConfig:
    def test_as_dict(self):
        cfg = InstrumentationConfig(
            targets=[InstrumentationTarget(module_name="m")],
            mode="exclude",
        )
        d = cfg.as_dict()
        assert d["mode"] == "exclude"
        assert len(d["targets"]) == 1

    def test_from_dict(self):
        d = {
            "targets": [{"module_name": "a"}, {"module_name": "b", "class_name": "B"}],
            "mode": "include",
        }
        cfg = InstrumentationConfig.from_dict(d)
        assert cfg.mode == "include"
        assert len(cfg.targets) == 2

    def test_from_dict_defaults(self):
        cfg = InstrumentationConfig.from_dict({})
        assert cfg.targets == []
        assert cfg.mode == "include"

    def test_roundtrip(self):
        cfg = InstrumentationConfig(
            targets=[InstrumentationTarget("mod", "Cls", "meth")],
            mode="exclude",
        )
        cfg2 = InstrumentationConfig.from_dict(cfg.as_dict())
        assert cfg2.mode == cfg.mode
        assert cfg2.targets[0].full_path == cfg.targets[0].full_path


# ---------------------------------------------------------------------------
# match_target
# ---------------------------------------------------------------------------


class TestMatchTarget:
    def test_exact_match(self):
        t = InstrumentationTarget("mod", "Cls", "run")
        assert match_target("mod.Cls.run", t) is True

    def test_partial_module_match(self):
        t = InstrumentationTarget("mod")
        assert match_target("mod.Cls.run", t) is True

    def test_partial_module_class_match(self):
        t = InstrumentationTarget("mod", "Cls")
        assert match_target("mod.Cls.run", t) is True

    def test_no_match(self):
        t = InstrumentationTarget("other")
        assert match_target("mod.Cls.run", t) is False

    def test_disabled_target_no_match(self):
        t = InstrumentationTarget("mod", "Cls", "run", enabled=False)
        assert match_target("mod.Cls.run", t) is False

    def test_no_false_prefix_match(self):
        t = InstrumentationTarget("mo")
        assert match_target("mod.Cls.run", t) is False


# ---------------------------------------------------------------------------
# should_instrument
# ---------------------------------------------------------------------------


class TestShouldInstrument:
    def test_include_mode_match(self):
        cfg = create_include_config(["mod.Cls.run"])
        assert should_instrument("mod.Cls.run", cfg) is True

    def test_include_mode_no_match(self):
        cfg = create_include_config(["mod.Cls.run"])
        assert should_instrument("other.Cls.run", cfg) is False

    def test_exclude_mode_match(self):
        cfg = create_exclude_config(["mod.Cls.run"])
        assert should_instrument("mod.Cls.run", cfg) is False

    def test_exclude_mode_no_match(self):
        cfg = create_exclude_config(["mod.Cls.run"])
        assert should_instrument("other.Cls.run", cfg) is True

    def test_empty_include_instruments_nothing(self):
        cfg = InstrumentationConfig(targets=[], mode="include")
        assert should_instrument("anything", cfg) is False

    def test_empty_exclude_instruments_everything(self):
        cfg = InstrumentationConfig(targets=[], mode="exclude")
        assert should_instrument("anything", cfg) is True


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


class TestFactories:
    def test_create_include_config(self):
        cfg = create_include_config(["mod.Cls.run", "other"])
        assert cfg.mode == "include"
        assert len(cfg.targets) == 2
        assert cfg.targets[0].full_path == "mod.Cls.run"
        assert cfg.targets[1].full_path == "other"

    def test_create_exclude_config(self):
        cfg = create_exclude_config(["mod.Cls"])
        assert cfg.mode == "exclude"
        assert cfg.targets[0].class_name == "Cls"

    def test_parse_single_part(self):
        cfg = create_include_config(["mymodule"])
        t = cfg.targets[0]
        assert t.module_name == "mymodule"
        assert t.class_name == ""
        assert t.method_name == ""

    def test_parse_two_parts(self):
        cfg = create_include_config(["mod.Cls"])
        t = cfg.targets[0]
        assert t.module_name == "mod"
        assert t.class_name == "Cls"
        assert t.method_name == ""

    def test_parse_three_parts(self):
        cfg = create_include_config(["mod.Cls.meth"])
        t = cfg.targets[0]
        assert t.module_name == "mod"
        assert t.class_name == "Cls"
        assert t.method_name == "meth"


# ---------------------------------------------------------------------------
# list_instrumented_targets
# ---------------------------------------------------------------------------


class TestListInstrumentedTargets:
    def test_include_mode(self):
        cfg = create_include_config(["mod"])
        all_targets = ["mod.A.run", "mod.B.go", "other.C.do"]
        result = list_instrumented_targets(cfg, all_targets)
        assert result == ["mod.A.run", "mod.B.go"]

    def test_exclude_mode(self):
        cfg = create_exclude_config(["mod.A"])
        all_targets = ["mod.A.run", "mod.B.go", "other.C.do"]
        result = list_instrumented_targets(cfg, all_targets)
        assert result == ["mod.B.go", "other.C.do"]

    def test_empty_all_targets(self):
        cfg = create_include_config(["mod"])
        assert list_instrumented_targets(cfg, []) == []
