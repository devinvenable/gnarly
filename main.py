#!/usr/bin/env python3
"""
Cellular Automata with Deep Dream Integration

This script processes an input image or video file with a combination of cellular
automata and Deep Dream effects. The output is displayed in a Pygame window and
recorded to an output video file.

Usage:
    python script.py input_file [--skip-face-detection] [--clear-cache]
"""

import argparse
import sys
import os
import random

import cv2
import numpy as np
import pygame
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.models import Inception_V3_Weights
import dlib  # For facial landmark detection (currently not used)

# ==== Configuration Constants ====
FPS = 15  # Frames per second for display and video output
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRID_SCALE = 4  # Number of image pixels per grid cell

DIVISOR_CHANGE_INTERVAL = 20  # Change divisor every 20 iterations (not implemented yet)
divisor = 2  # Initial divisor value for DivisorRule


# ==== Utility Functions ====

def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk and convert it to a numpy array."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        print(f"Loaded image '{image_path}' with shape {image_array.shape}")
        return image_array
    except Exception as e:
        raise Exception(f"Error loading image '{image_path}': {e}")


def initialize_grid(width: int, height: int) -> np.ndarray:
    """
    Initialize a grid for cellular automata.
    The grid is divided into 8x8 blocks, each randomly set as alive (1) or dead (0).
    """
    grid = np.zeros((height, width), dtype=int)
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            if random.random() < 0.5:
                grid[y:y+8, x:x+8] = 1
    return grid


def init_game_of_life_grid(grid_shape: tuple) -> np.ndarray:
    """Initialize the Game of Life grid with custom settings."""
    height, width = grid_shape
    grid = initialize_grid(width, height)
    print(f"Initialized Game of Life grid with shape {grid.shape}")
    return grid


def update_grid(grid: np.ndarray, rule: str, divisor_val: int) -> np.ndarray:
    """
    Compute the next generation of the grid based on the specified rule.

    Supported rules:
        - 'Conway': Conway's Game of Life
        - 'HighLife': HighLife variant (birth with 3 or 6 neighbors)
        - 'Seeds': Cells die every generation; birth at 2 neighbors
        - 'Custom': Survival with 2-4 neighbors; birth at 3 neighbors
        - 'DivisorRule': Math-based rule using a divisor
    """
    # Compute neighbor count with a toroidal (wrap-around) approach.
    neighbors = sum(np.roll(np.roll(grid, i, axis=0), j, axis=1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if not (i == 0 and j == 0))

    if rule == 'Conway':
        new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)
        new_grid = np.where((grid == 0) & (neighbors == 3), 1, new_grid)
    elif rule == 'HighLife':
        new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)
        new_grid = np.where((grid == 0) & ((neighbors == 3) | (neighbors == 6)), 1, new_grid)
    elif rule == 'Seeds':
        new_grid = np.zeros_like(grid)
        new_grid = np.where((grid == 0) & (neighbors == 2), 1, new_grid)
    elif rule == 'Custom':
        new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 4)), 0, grid)
        new_grid = np.where((grid == 0) & (neighbors == 3), 1, new_grid)
    elif rule == 'DivisorRule':
        remainder = neighbors % divisor_val
        new_value = remainder / divisor_val
        new_grid = np.where(new_value >= 0.5, 1, 0)
    else:
        # Default to Conway's rules if an unknown rule is provided.
        new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)
        new_grid = np.where((grid == 0) & (neighbors == 3), 1, new_grid)

    return new_grid


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Blend two images using the provided alpha mask.

    Args:
        img1: First image as a numpy array.
        img2: Second image as a numpy array (if None, returns img1).
        alpha: Alpha mask with values between 0 and 1.

    Returns:
        Blended image as a numpy array.
    """
    try:
        if img2 is None:
            return img1.astype(np.uint8)
        if img1.shape != img2.shape:
            raise ValueError(f"Shape mismatch: img1 {img1.shape} vs img2 {img2.shape}")
        if alpha.ndim == 2:
            alpha = alpha[:, :, np.newaxis]
        blended = img1.astype(np.float32) * (1 - alpha) + img2.astype(np.float32) * alpha
        blended = np.clip(blended, 0, 255)
        return blended.astype(np.uint8)
    except Exception as e:
        print(f"Error blending images: {e}")
        sys.exit(1)


# ==== Deep Dream Functions ====

def preprocess_image_for_deep_dream(image: Image.Image) -> torch.Tensor:
    """
    Preprocess the image for Deep Dream processing.

    Args:
        image: Input PIL Image.

    Returns:
        Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return image_tensor


def postprocess_deep_dream_output(tensor: torch.Tensor) -> np.ndarray:
    """
    Post-process the Deep Dream output tensor to a numpy array.

    Denormalizes the tensor and converts it back to an image.
    """
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    return np.array(image)


def load_deep_dream_model() -> torch.nn.Module:
    """
    Load the pre-trained InceptionV3 model for Deep Dream processing.

    Returns:
        InceptionV3 model in evaluation mode.
    """
    print("Loading InceptionV3 model for Deep Dream...")
    try:
        model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT,
                                    aux_logits=True).to(DEVICE).eval()
        print("InceptionV3 model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading InceptionV3 model: {e}")
        sys.exit(1)


def deep_dream_processing(image: np.ndarray, model: torch.nn.Module, layers: list,
                            iterations: int = 20, lr: float = 0.02) -> np.ndarray:
    """
    Apply Deep Dream processing to the image.

    Args:
        image: Input image as a numpy array.
        model: Pre-trained model for Deep Dream.
        layers: List of layer names to enhance.
        iterations: Number of optimization iterations.
        lr: Learning rate for optimization.

    Returns:
        Processed image as a numpy array.
    """
    try:
        input_tensor = preprocess_image_for_deep_dream(Image.fromarray(image))
        activations = []

        def hook_fn(module, input, output):
            activations.append(output)

        # Register hooks on specified layers
        hooks = []
        for name, module in model.named_modules():
            if name in layers:
                hooks.append(module.register_forward_hook(hook_fn))

        if not hooks:
            print("Warning: No hooks registered. Check provided layer names.")

        input_tensor.requires_grad_(True)
        optimizer = torch.optim.Adam([input_tensor], lr=lr, weight_decay=1e-4)

        for i in range(iterations):
            optimizer.zero_grad()
            activations.clear()  # Clear previous activations
            output = model(input_tensor)
            # If model returns a tuple (due to aux_logits), use only the primary output.
            if isinstance(output, tuple):
                _ = output[0]
            else:
                _ = output
            if activations:
                loss = torch.stack([act.norm() for act in activations]).sum()
                loss.backward()
                optimizer.step()
                if (i + 1) % 5 == 0:
                    print(f"Deep Dream Iteration {i+1}/{iterations}, Loss: {loss.item():.4f}")
            else:
                print("No activations collected; skipping optimization step.")
                break

        # Remove hooks and post-process the output.
        for hook in hooks:
            hook.remove()

        output_image = postprocess_deep_dream_output(input_tensor)
        return output_image

    except Exception as e:
        print(f"Error during deep dream processing: {e}")
        return image  # Return original image on error


def print_model_layers(model: torch.nn.Module) -> None:
    """Print all layer names of the model."""
    print("Model layers:")
    for name, _ in model.named_modules():
        print(name)


# ==== Main Functionality ====

def is_video_file(filename: str) -> bool:
    """Determine if the file is a video based on its extension."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def main(content_input_path: str, skip_face_detection: bool = False,
         clear_cache: bool = False) -> None:
    """
    Process the input image or video using cellular automata and Deep Dream effects.

    Args:
        content_input_path: Path to the content image or video.
        skip_face_detection: Flag to skip face detection (not implemented).
        clear_cache: Flag to clear cache before processing (not implemented).
    """
    global divisor  # Using the global divisor value

    cap = None  # Video capture handle (if needed)
    content_image_array = None

    try:
        # Load input (image or video)
        if is_video_file(content_input_path):
            cap = cv2.VideoCapture(content_input_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file {content_input_path}")
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video opened: {frame_count} frames at {input_fps} FPS")
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to read the first frame of the video.")
            content_image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            content_image_array = load_image(content_input_path)
            frame_count = None

        if content_image_array is None:
            raise Exception(f"Failed to load content from {content_input_path}")

        # Resize image if too large
        original_height, original_width, _ = content_image_array.shape
        max_dimension = 512
        scaling_factor = max_dimension / max(original_height, original_width)
        if scaling_factor < 1:
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            content_image_array = cv2.resize(content_image_array, (new_width, new_height),
                                             interpolation=cv2.INTER_AREA)
            height, width = content_image_array.shape[:2]
            print(f"Resized image to {width}x{height}")
        else:
            height, width = original_width, original_height
            print(f"Image size is {width}x{height}; no resizing needed.")

        # Pad image so dimensions are divisible by GRID_SCALE
        pad_height = (GRID_SCALE - height % GRID_SCALE) if height % GRID_SCALE != 0 else 0
        pad_width = (GRID_SCALE - width % GRID_SCALE) if width % GRID_SCALE != 0 else 0
        if pad_height or pad_width:
            content_image_array = cv2.copyMakeBorder(
                content_image_array, 0, pad_height, 0, pad_width, cv2.BORDER_REPLICATE
            )
            height, width = content_image_array.shape[:2]
            print(f"Padded image to {width}x{height} for GRID_SCALE={GRID_SCALE}")
        else:
            print("No padding needed for image dimensions.")

        # Initialize the cumulative Deep Dream image (for feedback effect)
        cumulative_dream_image = content_image_array.copy()

        # Load Deep Dream model and list layers
        deep_dream_model = load_deep_dream_model()
        print_model_layers(deep_dream_model)
        dream_layers = ['Mixed_7b', 'Mixed_7c', 'Mixed_7c.branch3x3_2a.conv']
        deep_dream_iterations = 20
        deep_dream_lr = 0.16

        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()
        print(f"Pygame window initialized with size {width}x{height}")

        # Initialize VideoWriter for output video
        video_filename = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = input_fps if cap is not None else FPS
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
        print(f"Video writer initialized: '{video_filename}'")

        # Initialize cellular automata grid
        grid_height = height // GRID_SCALE
        grid_width = width // GRID_SCALE
        grid_shape = (grid_height, grid_width)
        grid = init_game_of_life_grid(grid_shape)

        # Set initial cellular automata rule
        rules = ['Conway', 'HighLife', 'Seeds', 'Custom', 'DivisorRule']
        rule_index = 0
        current_rule = rules[rule_index]
        pygame.display.set_caption(f"Cellular Automata Rule: {current_rule}")
        print(f"Initial cellular automata rule: {current_rule}")

        previous_output = None  # For echo effect in blending

        # Main loop
        running = True
        while running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        rule_index = (rule_index + 1) % len(rules)
                        current_rule = rules[rule_index]
                        pygame.display.set_caption(f"Cellular Automata Rule: {current_rule}")
                        print(f"Switched cellular automata rule to: {current_rule}")
                    elif event.key == pygame.K_r:
                        grid = init_game_of_life_grid(grid_shape)
                        if cap is None:
                            content_image_array = load_image(content_input_path)
                            cumulative_dream_image = content_image_array.copy()
                        else:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                raise Exception("Failed to read the first frame after reset.")
                            content_image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            content_image_array = cv2.resize(content_image_array, (width, height))
                            cumulative_dream_image = content_image_array.copy()
                        previous_output = None
                        print("Reset grid and image.")

            # Read new frame if input is a video
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached.")
                    running = False
                    break
                content_image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                content_image_array = cv2.resize(content_image_array, (width, height))
                cumulative_dream_image = content_image_array.copy()

            # Optionally blend with previous output to create an echo effect
            if previous_output is not None:
                alpha = 0.5  # Blending factor
                content_image_array = cv2.addWeighted(content_image_array, alpha,
                                                      previous_output, 1 - alpha, 0)

            # Update cellular automata grid
            grid = update_grid(grid, current_rule, divisor)
            print(f"Updated grid using rule '{current_rule}'")

            # Reinitialize grid if all cells are dead
            if np.sum(grid) == 0:
                grid = init_game_of_life_grid(grid_shape)
                print("Grid died out; reinitialized.")

            # Upscale grid to image resolution
            upscaled_grid = cv2.resize(grid.astype(np.float32), (width, height),
                                       interpolation=cv2.INTER_NEAREST)
            alive_mask = upscaled_grid  # Alive cells mask
            dead_mask = 1 - upscaled_grid  # Dead cells mask

            # Create composite image using grid masks to blend current content and cumulative dream image
            composite_image = (
                content_image_array.astype(np.float32) * alive_mask[:, :, np.newaxis] +
                cumulative_dream_image.astype(np.float32) * dead_mask[:, :, np.newaxis]
            ).astype(np.uint8)

            # Apply Deep Dream processing to the composite image
            deep_dreamed_image = deep_dream_processing(
                composite_image,
                model=deep_dream_model,
                layers=dream_layers,
                iterations=deep_dream_iterations,
                lr=deep_dream_lr
            )
            cumulative_dream_image = deep_dreamed_image.copy()
            previous_output = content_image_array.copy()

            # Display the output using Pygame
            try:
                surface = pygame.surfarray.make_surface(np.swapaxes(cumulative_dream_image, 0, 1))
            except Exception as e:
                print(f"Error creating Pygame surface: {e}")
                sys.exit(1)
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Write the frame to the output video (convert from RGB to BGR)
            try:
                output_frame = cv2.cvtColor(cumulative_dream_image, cv2.COLOR_RGB2BGR)
                video_writer.write(output_frame)
            except Exception as e:
                print(f"Error writing frame to video: {e}")
                sys.exit(1)

            clock.tick(FPS)

        # Cleanup resources
        if cap is not None:
            cap.release()
        video_writer.release()
        pygame.quit()
        print(f"Video saved as '{video_filename}'")
        sys.exit(0)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cellular Automata with Deep Dream.')
    parser.add_argument('input_file', type=str, help='Path to the content image or video file.')
    parser.add_argument('--skip-face-detection', action='store_true',
                        help='Skip face detection (not implemented).')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cache before processing (not implemented).')
    args = parser.parse_args()

    main(args.input_file, skip_face_detection=args.skip_face_detection,
         clear_cache=args.clear_cache)
