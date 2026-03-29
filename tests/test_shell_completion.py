"""Tests for shell completion script generation."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.shell_completion import (
    CompletionConfig,
    generate_bash_completion,
    generate_zsh_completion,
    generate_fish_completion,
    generate_completion,
    get_evalyn_commands,
    get_evalyn_flags,
    format_install_instructions,
)


# ---------------------------------------------------------------------------
# CompletionConfig dataclass
# ---------------------------------------------------------------------------


class TestCompletionConfig:
    def test_defaults(self):
        cfg = CompletionConfig(shell="bash")
        assert cfg.shell == "bash"
        assert cfg.commands == []
        assert cfg.flags == {}

    def test_full_init(self):
        cfg = CompletionConfig(
            shell="zsh",
            commands=["run", "test"],
            flags={"run": ["--verbose"]},
        )
        assert cfg.shell == "zsh"
        assert cfg.commands == ["run", "test"]
        assert cfg.flags == {"run": ["--verbose"]}

    def test_as_dict(self):
        cfg = CompletionConfig(
            shell="fish",
            commands=["a"],
            flags={"a": ["--help"]},
        )
        d = cfg.as_dict()
        assert d == {
            "shell": "fish",
            "commands": ["a"],
            "flags": {"a": ["--help"]},
        }

    def test_from_dict(self):
        d = {"shell": "bash", "commands": ["x"], "flags": {"x": ["--y"]}}
        cfg = CompletionConfig.from_dict(d)
        assert cfg.shell == "bash"
        assert cfg.commands == ["x"]
        assert cfg.flags == {"x": ["--y"]}

    def test_from_dict_defaults(self):
        cfg = CompletionConfig.from_dict({"shell": "zsh"})
        assert cfg.commands == []
        assert cfg.flags == {}

    def test_roundtrip(self):
        cfg = CompletionConfig(
            shell="bash",
            commands=["a", "b"],
            flags={"a": ["--x"], "b": ["--y", "--z"]},
        )
        cfg2 = CompletionConfig.from_dict(cfg.as_dict())
        assert cfg2.shell == cfg.shell
        assert cfg2.commands == cfg.commands
        assert cfg2.flags == cfg.flags

    def test_as_dict_deep_copies_lists(self):
        original = ["--flag"]
        cfg = CompletionConfig(shell="bash", commands=["cmd"], flags={"cmd": original})
        d = cfg.as_dict()
        d["flags"]["cmd"].append("--extra")
        assert cfg.flags["cmd"] == ["--flag"]

    def test_from_dict_deep_copies(self):
        d = {"shell": "bash", "commands": ["cmd"], "flags": {"cmd": ["--f"]}}
        cfg = CompletionConfig.from_dict(d)
        d["commands"].append("extra")
        d["flags"]["cmd"].append("--extra")
        assert cfg.commands == ["cmd"]
        assert cfg.flags == {"cmd": ["--f"]}


# ---------------------------------------------------------------------------
# get_evalyn_commands / get_evalyn_flags
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_commands_returns_list(self):
        cmds = get_evalyn_commands()
        assert isinstance(cmds, list)
        assert len(cmds) > 15

    def test_commands_contains_known(self):
        cmds = get_evalyn_commands()
        for expected in ["analyze", "run-eval", "export", "dashboard", "status"]:
            assert expected in cmds

    def test_commands_sorted(self):
        cmds = get_evalyn_commands()
        assert cmds == sorted(cmds)

    def test_flags_returns_dict(self):
        flags = get_evalyn_flags()
        assert isinstance(flags, dict)
        assert len(flags) > 5

    def test_flags_values_are_lists(self):
        flags = get_evalyn_flags()
        for cmd, flist in flags.items():
            assert isinstance(flist, list), f"{cmd} flags not a list"
            for f in flist:
                assert f.startswith("--"), f"Flag {f!r} for {cmd} missing --"

    def test_flags_returns_copy(self):
        f1 = get_evalyn_flags()
        f2 = get_evalyn_flags()
        f1["analyze"].append("--bogus")
        assert "--bogus" not in f2.get("analyze", [])

    def test_commands_returns_copy(self):
        c1 = get_evalyn_commands()
        c2 = get_evalyn_commands()
        c1.append("bogus")
        assert "bogus" not in c2


# ---------------------------------------------------------------------------
# Bash completion
# ---------------------------------------------------------------------------


class TestBashCompletion:
    def test_contains_complete_directive(self):
        script = generate_bash_completion(["run"], {}, "evalyn")
        assert "complete -F _evalyn_completions evalyn" in script

    def test_contains_function_name(self):
        script = generate_bash_completion(["run"], {})
        assert "_evalyn_completions()" in script

    def test_contains_commands(self):
        script = generate_bash_completion(["analyze", "export"], {})
        assert "analyze" in script
        assert "export" in script

    def test_contains_flags_in_case(self):
        flags = {"run": ["--verbose", "--output"]}
        script = generate_bash_completion(["run"], flags)
        assert "--verbose" in script
        assert "--output" in script

    def test_custom_prog_name(self):
        script = generate_bash_completion(["run"], {}, prog="myapp")
        assert "_myapp_completions" in script
        assert "complete -F _myapp_completions myapp" in script

    def test_empty_commands(self):
        script = generate_bash_completion([], {})
        assert "complete -F" in script

    def test_multiple_commands_in_case(self):
        flags = {"a": ["--x"], "b": ["--y"]}
        script = generate_bash_completion(["a", "b"], flags)
        assert "a)" in script
        assert "b)" in script

    def test_header_comment(self):
        script = generate_bash_completion(["run"], {})
        assert "# Bash completion for evalyn" in script


# ---------------------------------------------------------------------------
# Zsh completion
# ---------------------------------------------------------------------------


class TestZshCompletion:
    def test_compdef_directive(self):
        script = generate_zsh_completion(["run"], {})
        assert "#compdef evalyn" in script

    def test_function_name(self):
        script = generate_zsh_completion(["run"], {})
        assert "_evalyn()" in script

    def test_commands_listed(self):
        script = generate_zsh_completion(["analyze", "export"], {})
        assert "'analyze:" in script
        assert "'export:" in script

    def test_flags_in_case(self):
        flags = {"run": ["--verbose"]}
        script = generate_zsh_completion(["run"], flags)
        assert "'--verbose'" in script

    def test_custom_prog(self):
        script = generate_zsh_completion(["run"], {}, prog="myapp")
        assert "#compdef myapp" in script
        assert "_myapp()" in script

    def test_describe_command(self):
        script = generate_zsh_completion(["status"], {})
        assert "_describe 'command' commands" in script

    def test_trailing_call(self):
        script = generate_zsh_completion(["run"], {}, prog="evalyn")
        lines = script.strip().split("\n")
        assert lines[-1] == "_evalyn"


# ---------------------------------------------------------------------------
# Fish completion
# ---------------------------------------------------------------------------


class TestFishCompletion:
    def test_subcommand_completions(self):
        script = generate_fish_completion(["run", "test"], {})
        assert "__fish_use_subcommand" in script
        assert "-a 'run'" in script
        assert "-a 'test'" in script

    def test_flag_completions(self):
        flags = {"run": ["--verbose"]}
        script = generate_fish_completion(["run"], flags)
        assert "__fish_seen_subcommand_from run" in script
        assert "-l 'verbose'" in script

    def test_custom_prog(self):
        script = generate_fish_completion(["run"], {}, prog="myapp")
        assert "complete -c myapp" in script

    def test_header_comment(self):
        script = generate_fish_completion(["run"], {})
        assert "# Fish completion for evalyn" in script

    def test_multiple_flags(self):
        flags = {"run": ["--verbose", "--output"]}
        script = generate_fish_completion(["run"], flags)
        assert "-l 'verbose'" in script
        assert "-l 'output'" in script


# ---------------------------------------------------------------------------
# generate_completion dispatcher
# ---------------------------------------------------------------------------


class TestGenerateCompletion:
    def test_bash_dispatch(self):
        cfg = CompletionConfig(shell="bash", commands=["run"], flags={})
        result = generate_completion(cfg)
        assert "complete -F" in result

    def test_zsh_dispatch(self):
        cfg = CompletionConfig(shell="zsh", commands=["run"], flags={})
        result = generate_completion(cfg)
        assert "#compdef" in result

    def test_fish_dispatch(self):
        cfg = CompletionConfig(shell="fish", commands=["run"], flags={})
        result = generate_completion(cfg)
        assert "__fish_use_subcommand" in result

    def test_unsupported_shell_raises(self):
        cfg = CompletionConfig(shell="powershell", commands=[], flags={})
        try:
            generate_completion(cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "powershell" in str(e)

    def test_custom_prog(self):
        cfg = CompletionConfig(shell="bash", commands=["x"], flags={})
        result = generate_completion(cfg, prog="custom")
        assert "_custom_completions" in result

    def test_full_config_roundtrip(self):
        cmds = get_evalyn_commands()
        flags = get_evalyn_flags()
        for shell in ("bash", "zsh", "fish"):
            cfg = CompletionConfig(shell=shell, commands=cmds, flags=flags)
            result = generate_completion(cfg)
            assert len(result) > 100


# ---------------------------------------------------------------------------
# format_install_instructions
# ---------------------------------------------------------------------------


class TestInstallInstructions:
    def test_bash_instructions(self):
        text = format_install_instructions("bash")
        assert "bashrc" in text
        assert "evalyn" in text

    def test_zsh_instructions(self):
        text = format_install_instructions("zsh")
        assert "zshrc" in text

    def test_fish_instructions(self):
        text = format_install_instructions("fish")
        assert "fish" in text

    def test_custom_prog(self):
        text = format_install_instructions("bash", prog="myapp")
        assert "myapp" in text

    def test_unsupported_shell(self):
        try:
            format_install_instructions("tcsh")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "tcsh" in str(e)

    def test_zsh_mentions_compinit(self):
        text = format_install_instructions("zsh")
        assert "compinit" in text

    def test_fish_mentions_completions_dir(self):
        text = format_install_instructions("fish")
        assert "completions" in text
