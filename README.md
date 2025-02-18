# Cellular Automata with Deep Dream and Object Morphing

A Python-based creative visualization tool that combines cellular automata, Deep Dream effects, continuous zoom, and object morphing to create mesmerizing visual experiences from images and videos.

## Features

- Cellular automata-based image processing
- Deep Dream integration for enhanced visual effects
- Continuous zoom effects
- Object detection and morphing using YOLO and EfficientDet
- Support for both image and video input
- GPU acceleration support (CUDA)
- Stable Diffusion integration for object generation
- Face detection and landmark recognition

## Requirements

- Python 3.x
- CUDA-capable GPU (recommended)
- See `requirements.txt` for Python package dependencies

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Basic usage:
```bash
python main.py input_file [--output_file OUTPUT_FILE] [--deep_dream_iterations ITERATIONS] [--deep_dream_lr LEARNING_RATE] [--creativity CREATIVITY_LEVEL]
```

### Arguments

- `input_file`: Path to the input image or video file
- `--output_file`: (Optional) Path to the output video file. Defaults to input_name_dream.mp4 in keep/
- `--deep_dream_iterations`: (Optional) Number of Deep Dream iterations (default: 30)
- `--deep_dream_lr`: (Optional) Learning rate for Deep Dream (default: 0.02)
- `--creativity`: (Optional) Creativity level for object detection/morphing (0.0 to 1.0, default: 0.5)

### Interactive Controls

During rendering, you can use the following keyboard controls to adjust parameters in real-time:

- `Space`: Switch between different cellular automata rules
- `R`: Reset the grid, image, and zoom settings
- `Left/Right Arrow`: Decrease/Increase grid scale
- `Up/Down Arrow`: Increase/Decrease the divisor value
- `B/N`: Increase/Decrease blend alpha (mixing between states)
- `O`: Toggle object detection and morphing
- `Z`: Toggle continuous zoom effect
- `[/]`: Decrease/Increase creativity level
- `Esc`: Exit the program

## How It Works

1. The program processes input media using cellular automata rules
2. Deep Dream effects are applied to enhance visual patterns
3. Object detection identifies interesting elements in the scene
4. Stable Diffusion generates variations of detected objects
5. Continuous zoom creates an infinite zoom effect
6. All elements are combined into a seamless, evolving visual experience

## Credits

This project was developed by Devin Venable in collaboration with AI assistance.

## Video Examples

Here are all example outputs from the cellular automata processing:

### Human Portraits Series
[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/human_24_dream_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/human_24_dream.mp4)

[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/human_26_dream_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/human_26_dream.mp4)

[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/human_28_dream_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/human_28_dream.mp4)

[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/human_29_dream_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/human_29_dream.mp4)

[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/human_9_dream_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/human_9_dream.mp4)

### Nature and Landscapes
[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/flower_5_dream_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/flower_5_dream.mp4)

[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/leaves_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/leaves.mp4)

### Abstract and Stylized
[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/style_15_dream_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/style_15_dream.mp4)

[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/style_24_dream_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/style_24_dream.mp4)

### Other Examples
[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/one_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/one.mp4)

[![Watch the video](https://cellular-automata-videos.s3.amazonaws.com/UK_NewYork_US_Header_dream_thumb.jpg)](https://cellular-automata-videos.s3.amazonaws.com/UK_NewYork_US_Header_dream.mp4)

## License

This project is open source and available under the MIT License.
