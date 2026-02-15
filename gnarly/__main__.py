"""Entry point for python -m gnarly."""

from .cli import parse_args
from .runners.headless import run_headless


def main():
    """Main entry point."""
    config = parse_args()
    run_headless(config)


if __name__ == "__main__":
    main()
