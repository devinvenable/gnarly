"""Command-line interface for gnarly."""

import argparse
from pathlib import Path

from . import __version__
from .config import ProcessingConfig
from .core.ca import CAEngine
from .effects.morph_effect import STYLE_NAMES

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

Use `--deep-dream` to enable GPU Deep Dream processing.
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

    parser.add_argument(
        "--deep-dream",
        action="store_true",
        help="Enable Deep Dream effect (requires CUDA GPU)",
    )

    parser.add_argument(
        "--dream-iterations",
        type=int,
        default=30,
        help="Deep Dream optimization iterations per frame (default: 30)",
    )

    parser.add_argument(
        "--dream-lr",
        type=float,
        default=0.02,
        help="Deep Dream learning rate (default: 0.02)",
    )

    parser.add_argument(
        "--dream-layers",
        type=str,
        default="",
        help=(
            "Comma-separated Inception layer names for Deep Dream targets "
            "(default: built-in Mixed_7* layers)"
        ),
    )

    # Zoom effect arguments
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Enable continuous zoom effect",
    )

    parser.add_argument(
        "--zoom-speed",
        type=float,
        default=1.02,
        help="Zoom factor multiplier per frame; 1.02 = 2%% zoom/frame (default: 1.02)",
    )

    parser.add_argument(
        "--zoom-min",
        type=float,
        default=1.0,
        help="Minimum zoom level; 1.0 = no zoom (default: 1.0)",
    )

    parser.add_argument(
        "--zoom-max",
        type=float,
        default=2.0,
        help="Maximum zoom level before reversing (default: 2.0)",
    )

    # Detection arguments
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Enable object detection (requires CUDA GPU)",
    )

    parser.add_argument(
        "--detect-model",
        type=str,
        default="yolo",
        choices=["yolo"],
        help="Detection model to use (default: yolo)",
    )

    parser.add_argument(
        "--detect-confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold; 0.0-1.0 (default: 0.5)",
    )

    parser.add_argument(
        "--detect-interval",
        type=int,
        default=30,
        help="Run detection every N frames (default: 30)",
    )

    # Morph arguments
    parser.add_argument(
        "--morph",
        action="store_true",
        help="Enable object morphing effect (requires detector integration and CUDA)",
    )

    parser.add_argument(
        "--morph-style",
        type=str,
        default="random",
        choices=["random", *sorted(STYLE_NAMES)],
        help="Morph image style preset (default: random)",
    )

    parser.add_argument(
        "--creativity",
        type=float,
        default=0.5,
        help="Creativity level for morph prompt/guidance variation, 0.0-1.0 (default: 0.5)",
    )

    parsed = parser.parse_args(args)

    # Validate input exists
    if not Path(parsed.input).exists():
        parser.error(f"Input file not found: {parsed.input}")
    if not 0.0 <= parsed.creativity <= 1.0:
        parser.error("--creativity must be between 0.0 and 1.0")

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
        zoom_enabled=parsed.zoom,
        zoom_speed=parsed.zoom_speed,
        zoom_min=parsed.zoom_min,
        zoom_max=parsed.zoom_max,
        deep_dream_enabled=parsed.deep_dream,
        deep_dream_iterations=parsed.dream_iterations,
        deep_dream_lr=parsed.dream_lr,
        deep_dream_layers=(
            [layer.strip() for layer in parsed.dream_layers.split(",") if layer.strip()]
            if parsed.dream_layers
            else None
        ),
        detection_enabled=parsed.detect,
        detection_model=parsed.detect_model,
        detection_confidence=parsed.detect_confidence,
        detection_interval=parsed.detect_interval,
        morph_enabled=parsed.morph,
        morph_style=parsed.morph_style,
        morph_creativity=parsed.creativity,
    )
