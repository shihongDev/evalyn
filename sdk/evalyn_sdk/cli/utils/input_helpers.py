"""Input helper functions for interactive CLI commands."""

from __future__ import annotations

from typing import Optional


def truncate_text(text: str, max_len: int = 300) -> str:
    """Truncate text to max_len characters, replacing newlines with spaces.

    Args:
        text: Text to truncate
        max_len: Maximum length (default 300)

    Returns:
        Truncated text with ellipsis if needed
    """
    text = str(text) if text else ""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[:max_len] + "..."


def get_bool_input(prompt: str, allow_skip: bool = True) -> Optional[bool]:
    """Get yes/no input from user.

    Args:
        prompt: Question to display
        allow_skip: Whether to allow skipping (default True)

    Returns:
        True for yes, False for no, None for skip/cancel
    """
    suffix = " [y/n/s]: " if allow_skip else " [y/n]: "
    while True:
        try:
            val = input(f"  {prompt}{suffix}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if val in ("y", "yes", "1", "true"):
            return True
        if val in ("n", "no", "0", "false"):
            return False
        if allow_skip and val in ("s", "skip", ""):
            return None
        print(f"  Invalid. Use y(es), n(o){', or s(kip)' if allow_skip else ''}")


def get_int_input(
    prompt: str, min_val: int, max_val: int, allow_skip: bool = True
) -> Optional[int]:
    """Get integer input within range from user.

    Args:
        prompt: Question to display
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_skip: Whether to allow skipping (default True)

    Returns:
        Integer within range, or None for skip/cancel
    """
    suffix = f" [{min_val}-{max_val}/s]: " if allow_skip else f" [{min_val}-{max_val}]: "
    while True:
        try:
            val = input(f"  {prompt}{suffix}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if allow_skip and val in ("s", "skip", ""):
            return None
        try:
            num = int(val)
            if min_val <= num <= max_val:
                return num
            print(f"  Invalid. Use {min_val}-{max_val}{' or s(kip)' if allow_skip else ''}")
        except ValueError:
            print(f"  Invalid. Use {min_val}-{max_val}{' or s(kip)' if allow_skip else ''}")


def get_str_input(prompt: str) -> str:
    """Get string input from user.

    Args:
        prompt: Question to display

    Returns:
        User input string, or empty string on cancel
    """
    try:
        return input(f"  {prompt} ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def get_confidence(prompt: str = "Confidence (1-5, Enter to skip): ") -> Optional[int]:
    """Get confidence score 1-5 from user with recursive retry.

    Args:
        prompt: Question to display

    Returns:
        Confidence score 1-5, or None if skipped
    """
    try:
        conf_input = input(prompt).strip()
        if not conf_input:
            return None
        conf = int(conf_input)
        if 1 <= conf <= 5:
            return conf
        print("Invalid. Use 1-5.")
        return get_confidence(prompt)
    except ValueError:
        print("Invalid. Use 1-5 or Enter to skip.")
        return get_confidence(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


__all__ = [
    "truncate_text",
    "get_bool_input",
    "get_int_input",
    "get_str_input",
    "get_confidence",
]
