"""
Utility functions for the project.
"""


def print_warning(message: str) -> None:
    """Print warning message with orange color formatting."""
    print(f"⚠️\t\033[38;5;208m{message}\033[0m")


def print_error(message: str) -> None:
    """Print error message with red color formatting."""
    print(f"❌\t\033[91m{message}\033[0m")
