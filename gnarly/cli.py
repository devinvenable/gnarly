"""Command-line interface for gnarly."""

import argparse
from pathlib import Path

from . import __version__
from .config import ProcessingConfig
from .core.ca import CAEngine

EPILOG = """\
Examples:
  gnarly input.jpg -o output.mp4
  gnarly input.jpg -o output.mp4 --rule HighLife --frames 500
  gnarly video.mp4 -o processed.mp4 --blend-alpha 0.5 --grid-scale 2

CA Rules:
  Conway      Classic Game of Life (B3/S23) - balanced patterns
  HighLife    Birth also with 6 neighbors (B36/S23) - replicators
  Seeds       Explosive growth, cells die instantly (B2/S)
  Custom      Modified survival (B3/S234) - denser patterns
  DivisorRule Modulo-based rule using --divisor value

Note: This is the lightweight headless mode. No GPU required.
For Deep Dream + GPU features, use: python main.py
"""


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
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input image (.jpg, .png) or video (.mp4, .avi) file",
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
        help="Output frame count for image input; ignored for video (default: 300)",
    )

    parser.add_argument(
        "--rule",
        type=str,
        default="Conway",
        choices=CAEngine.RULES,
        metavar="RULE",
        help="CA rule: Conway, HighLife, Seeds, Custom, DivisorRule (default: Conway)",
    )

    parser.add_argument(
        "--grid-scale",
        type=int,
        default=4,
        help="CA cell size in pixels; smaller = finer detail, larger = blocky (default: 4)",
    )

    parser.add_argument(
        "--divisor",
        type=int,
        default=2,
        help="Divisor for DivisorRule; affects pattern density (default: 2)",
    )

    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.3,
        help="Opacity of CA overlay on original image; 0.0=invisible, 1.0=opaque (default: 0.3)",
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
        help="Scale down frames to this max width/height for faster processing (default: 800)",
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
