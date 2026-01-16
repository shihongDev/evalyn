"""Shell completion generation for evalyn CLI.

This module provides commands to generate shell completion scripts for bash, zsh, and fish.

Usage:
  evalyn completion bash >> ~/.bashrc
  evalyn completion zsh >> ~/.zshrc
  evalyn completion fish > ~/.config/fish/completions/evalyn.fish
"""

from __future__ import annotations

import argparse
import sys

# All evalyn subcommands
COMMANDS = [
    "help",
    "list-calls",
    "show-call",
    "show-trace",
    "show-projects",
    "list-runs",
    "show-run",
    "build-dataset",
    "simulate",
    "export-for-annotation",
    "export",
    "status",
    "validate",
    "analyze",
    "compare",
    "trend",
    "annotate",
    "import-annotations",
    "annotation-stats",
    "calibrate",
    "list-calibrations",
    "run-eval",
    "suggest-metrics",
    "select-metrics",
    "list-metrics",
    "init",
    "one-click",
    "completion",
]


def _bash_completion() -> str:
    """Generate bash completion script."""
    commands_str = " ".join(COMMANDS)
    return f'''# evalyn bash completion
# Add to ~/.bashrc: eval "$(evalyn completion bash)"

_evalyn_completions() {{
    local cur prev commands
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"

    commands="{commands_str}"

    if [[ ${{COMP_CWORD}} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${{commands}}" -- "${{cur}}") )
        return 0
    fi

    # Common flags for all commands
    local common_flags="--help -h --quiet -q"

    case "${{prev}}" in
        --format)
            COMPREPLY=( $(compgen -W "table json" -- "${{cur}}") )
            return 0
            ;;
        --dataset|--output|--metrics|--run)
            # File/directory completion
            COMPREPLY=( $(compgen -f -- "${{cur}}") )
            return 0
            ;;
        --project)
            # Could add project completion here
            return 0
            ;;
    esac

    # Command-specific flags
    case "${{COMP_WORDS[1]}}" in
        list-calls|list-runs)
            COMPREPLY=( $(compgen -W "${{common_flags}} --limit --format --project" -- "${{cur}}") )
            ;;
        show-call|show-trace|show-run)
            COMPREPLY=( $(compgen -W "${{common_flags}} --id --last --format" -- "${{cur}}") )
            ;;
        run-eval)
            COMPREPLY=( $(compgen -W "${{common_flags}} --dataset --latest --metrics --workers --format" -- "${{cur}}") )
            ;;
        analyze)
            COMPREPLY=( $(compgen -W "${{common_flags}} --run --dataset --latest --format" -- "${{cur}}") )
            ;;
        *)
            COMPREPLY=( $(compgen -W "${{common_flags}}" -- "${{cur}}") )
            ;;
    esac

    return 0
}}

complete -F _evalyn_completions evalyn
'''


def _zsh_completion() -> str:
    """Generate zsh completion script."""
    commands_str = " ".join(COMMANDS)
    return f"""#compdef evalyn
# evalyn zsh completion
# Add to ~/.zshrc: eval "$(evalyn completion zsh)"

_evalyn() {{
    local -a commands
    commands=(
        'help:Show available commands'
        'list-calls:List captured function calls'
        'show-call:Show details of a specific call'
        'show-trace:Show hierarchical span tree'
        'show-projects:Show project summary'
        'list-runs:List stored eval runs'
        'show-run:Show details for an eval run'
        'build-dataset:Build dataset from traces'
        'simulate:Generate synthetic test data'
        'export:Export results'
        'export-for-annotation:Export for annotation tools'
        'status:Show dataset status'
        'validate:Validate dataset format'
        'analyze:Analyze evaluation results'
        'compare:Compare two eval runs'
        'trend:Show evaluation trends'
        'annotate:Interactive annotation'
        'import-annotations:Import annotations'
        'annotation-stats:Show annotation statistics'
        'calibrate:Calibrate LLM judges'
        'list-calibrations:List calibration records'
        'run-eval:Run evaluation'
        'suggest-metrics:Suggest metrics'
        'select-metrics:Select metrics interactively'
        'list-metrics:List available metrics'
        'init:Initialize configuration'
        'one-click:Run complete pipeline'
        'completion:Generate shell completions'
    )

    _arguments -C \\
        '(-h --help){{-h,--help}}[Show help]' \\
        '(-q --quiet){{-q,--quiet}}[Suppress hints]' \\
        '1: :->command' \\
        '*::arg:->args'

    case "$state" in
        command)
            _describe -t commands 'evalyn commands' commands
            ;;
        args)
            case $words[1] in
                list-calls|list-runs)
                    _arguments \\
                        '--limit[Max items]:number:' \\
                        '--format[Output format]:format:(table json)' \\
                        '--project[Project filter]:project:'
                    ;;
                show-call|show-trace|show-run)
                    _arguments \\
                        '--id[Record ID]:id:' \\
                        '--last[Show most recent]' \\
                        '--format[Output format]:format:(table json)'
                    ;;
                run-eval)
                    _arguments \\
                        '--dataset[Dataset path]:file:_files' \\
                        '--latest[Use latest dataset]' \\
                        '--metrics[Metrics file]:file:_files' \\
                        '--workers[Worker count]:number:' \\
                        '--format[Output format]:format:(table json)'
                    ;;
                analyze)
                    _arguments \\
                        '--run[Run ID]:id:' \\
                        '--dataset[Dataset path]:file:_files' \\
                        '--latest[Use latest dataset]' \\
                        '--format[Output format]:format:(table json)'
                    ;;
            esac
            ;;
    esac
}}

_evalyn "$@"
"""


def _fish_completion() -> str:
    """Generate fish completion script."""
    lines = [
        "# evalyn fish completion",
        "# Save to ~/.config/fish/completions/evalyn.fish",
        "",
        "# Disable file completion by default",
        "complete -c evalyn -f",
        "",
        "# Commands",
    ]

    cmd_descriptions = {
        "help": "Show available commands",
        "list-calls": "List captured function calls",
        "show-call": "Show details of a specific call",
        "show-trace": "Show hierarchical span tree",
        "show-projects": "Show project summary",
        "list-runs": "List stored eval runs",
        "show-run": "Show details for an eval run",
        "build-dataset": "Build dataset from traces",
        "simulate": "Generate synthetic test data",
        "export": "Export results",
        "export-for-annotation": "Export for annotation tools",
        "status": "Show dataset status",
        "validate": "Validate dataset format",
        "analyze": "Analyze evaluation results",
        "compare": "Compare two eval runs",
        "trend": "Show evaluation trends",
        "annotate": "Interactive annotation",
        "import-annotations": "Import annotations",
        "annotation-stats": "Show annotation statistics",
        "calibrate": "Calibrate LLM judges",
        "list-calibrations": "List calibration records",
        "run-eval": "Run evaluation",
        "suggest-metrics": "Suggest metrics",
        "select-metrics": "Select metrics interactively",
        "list-metrics": "List available metrics",
        "init": "Initialize configuration",
        "one-click": "Run complete pipeline",
        "completion": "Generate shell completions",
    }

    for cmd, desc in cmd_descriptions.items():
        lines.append(
            f'complete -c evalyn -n "__fish_use_subcommand" -a "{cmd}" -d "{desc}"'
        )

    lines.extend(
        [
            "",
            "# Global flags",
            'complete -c evalyn -s h -l help -d "Show help"',
            'complete -c evalyn -s q -l quiet -d "Suppress hints"',
            "",
            "# Common flags for multiple commands",
            'complete -c evalyn -n "__fish_seen_subcommand_from list-calls list-runs" -l limit -d "Max items"',
            'complete -c evalyn -n "__fish_seen_subcommand_from list-calls list-runs show-call show-run analyze" -l format -a "table json" -d "Output format"',
            'complete -c evalyn -n "__fish_seen_subcommand_from show-call show-trace show-run" -l id -d "Record ID"',
            'complete -c evalyn -n "__fish_seen_subcommand_from show-call show-trace show-run" -l last -d "Show most recent"',
            'complete -c evalyn -n "__fish_seen_subcommand_from run-eval analyze" -l dataset -d "Dataset path"',
            'complete -c evalyn -n "__fish_seen_subcommand_from run-eval analyze compare trend" -l latest -d "Use latest dataset"',
        ]
    )

    return "\n".join(lines)


def cmd_completion(args: argparse.Namespace) -> None:
    """Generate shell completion script."""
    shell = args.shell

    if shell == "bash":
        print(_bash_completion())
    elif shell == "zsh":
        print(_zsh_completion())
    elif shell == "fish":
        print(_fish_completion())
    else:
        print(f"Unsupported shell: {shell}", file=sys.stderr)
        print("Supported shells: bash, zsh, fish", file=sys.stderr)
        sys.exit(1)


def register_commands(subparsers) -> None:
    """Register completion command."""
    p = subparsers.add_parser(
        "completion",
        help="Generate shell completion script",
        description="Generate shell completion scripts for bash, zsh, or fish.",
        epilog="""
Examples:
  evalyn completion bash >> ~/.bashrc
  evalyn completion zsh >> ~/.zshrc
  evalyn completion fish > ~/.config/fish/completions/evalyn.fish

  # Or use eval for immediate effect:
  eval "$(evalyn completion bash)"
""",
    )
    p.add_argument(
        "shell",
        choices=["bash", "zsh", "fish"],
        help="Shell to generate completion for",
    )
    p.set_defaults(func=cmd_completion)


__all__ = ["cmd_completion", "register_commands"]
