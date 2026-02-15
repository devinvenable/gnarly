"""Command-line interface for gnarly."""

import argparse
from pathlib import Path

from .config import ProcessingConfig
from .core.ca import CAEngine


def parse_args(args=None) -> ProcessingConfig:
    """Parse command-line arguments.

    Args:
        args: Arguments to parse (defaults to sys.argv).

    Returns:
        ProcessingConfig with parsed options.
    """
    parser = argparse.ArgumentParser(
        prog="gnarly",
        description="Apply cellular automata effects to images and videos.",
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input image or video file",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output video file",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Number of frames to generate (for images, default: 300)",
    )

    parser.add_argument(
        "--rule",
        type=str,
        default="Conway",
        choices=CAEngine.RULES,
        help="Cellular automata rule (default: Conway)",
    )

    parser.add_argument(
        "--grid-scale",
        type=int,
        default=4,
        help="CA cell size in pixels (default: 4)",
    )

    parser.add_argument(
        "--divisor",
        type=int,
        default=2,
        help="Divisor for DivisorRule (default: 2)",
    )

    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.3,
        help="Blend factor for live cells (default: 0.3)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Output frames per second (default: 15)",
    )

    parser.add_argument(
        "--max-dimension",
        type=int,
        default=800,
        help="Maximum frame dimension (default: 800)",
    )

    parsed = parser.parse_args(args)

    # Validate input exists
    if not Path(parsed.input).exists():
        parser.error(f"Input file not found: {parsed.input}")

    return ProcessingConfig.from_args(
        input_path=parsed.input,
        output_path=parsed.output,
        rule=parsed.rule,
        grid_scale=parsed.grid_scale,
        divisor=parsed.divisor,
        blend_alpha=parsed.blend_alpha,
        fps=parsed.fps,
        frames=parsed.frames,
        max_dimension=parsed.max_dimension,
    )
