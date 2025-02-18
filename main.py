#!/usr/bin/env python3
"""
Cellular Automata with Deep Dream, Continuous Zoom, and Object Morphing

This script processes an input image or video file using a combination of cellular 
automata, Deep Dream effects, continuous zoom, and optional object detection/morphing.
...
"""

import argparse
import sys
import os
import random
import gc
import time
import pandas as pd

import cv2
import numpy as np
import pygame
import torch
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms, models
from torchvision.models import Inception_V3_Weights

from diffusers import StableDiffusionPipeline

# ==== Configuration Constants ====
FPS = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Creativity and Detection Parameters
CREATIVITY = 0.5  # Controls how flexible object detection and matching is (0.0 to 1.0)
BASE_CONFIDENCE_THRESHOLD = 0.1  # Base confidence threshold for object detection
BASE_IOU_THRESHOLD = 0.5  # Base threshold for considering objects similar

# Artistic style variations
ARTISTIC_STYLES = [
    {
        'name': 'photorealistic',
        'prompts': [
            'a photorealistic {obj} with intricate details',
            'a high-resolution photograph of a {obj}',
            'a detailed realistic {obj} in natural lighting'
        ],
        'guidance_scale': 7.0
    },
    {
        'name': 'surrealistic',
        'prompts': [
            'a surrealistic {obj} in the style of Salvador Dali',
            'a dreamlike surreal {obj} with impossible physics',
            'a melting surrealistic {obj} in a bizarre landscape'
        ],
        'guidance_scale': 9.0
    },
    {
        'name': 'impressionist',
        'prompts': [
            'an impressionist painting of a {obj} in the style of Monet',
            'a vibrant impressionist {obj} with visible brushstrokes',
            'a light-filled impressionist scene featuring a {obj}'
        ],
        'guidance_scale': 8.0
    },
    {
        'name': 'cyberpunk',
        'prompts': [
            'a cyberpunk {obj} with neon lights and chrome',
            'a futuristic {obj} in a high-tech cyberpunk setting',
            'a {obj} reimagined as a cyber-enhanced device'
        ],
        'guidance_scale': 8.5
    },
    {
        'name': 'abstract',
        'prompts': [
            'an abstract interpretation of a {obj} with geometric shapes',
            'a Kandinsky-style abstract painting of a {obj}',
            'a colorful abstract {obj} with bold forms'
        ],
        'guidance_scale': 9.0
    }
]

DEFAULT_GRID_SCALE = 4      # More blocky default from earlier version
DEFAULT_DIVISOR = 2
DEFAULT_BLEND_ALPHA = 0.3   # Lower alpha to emphasize grid pattern
DEEP_DREAM_ITERATIONS = 30  
DEEP_DREAM_LR = 0.02
DREAM_LAYERS = ['Mixed_7a.branch3x3_2a.conv', 'Mixed_7b.branch3x3_2b.conv', 'Mixed_7c.branch3x3_2c.conv']

OBJECT_DETECTION_INTERVAL = 5
MORPH_FRAMES = 60
MORPH_MIN_BLEND = 0.1
MORPH_MAX_BLEND = 0.7
MAX_OBJECTS_PER_FRAME = 3
MAX_OBJECT_SIZE_RATIO = 0.25

ZOOM_ENABLED = False
ZOOM_SPEED = 0.001
MIN_ZOOM = 1.0
MAX_ZOOM = 2.0

# ==== Hugging Face Predictor Download (for face detection) ====
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please install huggingface_hub with 'pip install huggingface_hub'")
    sys.exit(1)

def get_shape_predictor_path() -> str:
    predictor_filename = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_filename):
        print(f"'{predictor_filename}' not found. Downloading from Hugging Face Hub...")
        try:
            predictor_path = hf_hub_download(repo_id="public-data/dlib_face_landmark_model",
                                             filename=predictor_filename)
            print(f"Downloaded predictor to {predictor_path}")
        except Exception as e:
            print(f"Error downloading predictor: {e}")
            sys.exit(1)
    else:
        predictor_path = predictor_filename
    return predictor_path

def detect_faces_and_landmarks(image_array):
    import dlib
    detector = dlib.get_frontal_face_detector()
    predictor_path = get_shape_predictor_path()
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    boxes = []
    landmarks_list = []
    for rect in rects:
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        boxes.append([x1, y1, x2, y2])
        shape = predictor(gray, rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        landmarks_list.append(landmarks)
    print(f"Detected {len(landmarks_list)} face(s)")
    return np.array(boxes), landmarks_list

# ==== Cellular Automata Functions ====
def initialize_grid(width: int, height: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            if random.random() < 0.5:
                grid[y:y+8, x:x+8] = 1
    return grid

def init_game_of_life_grid(grid_shape: tuple) -> np.ndarray:
    height, width = grid_shape
    grid = initialize_grid(width, height)
    print(f"Initialized grid with shape {grid.shape}")
    return grid

def update_grid(grid: np.ndarray, rule: str, divisor_val: int) -> np.ndarray:
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
        new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)
        new_grid = np.where((grid == 0) & (neighbors == 3), 1, new_grid)
    return new_grid

def blend_images(img1, img2, alpha):
    try:
        if img2 is None:
            return img1.astype(np.uint8)
        if img1.shape != img2.shape:
            raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
        if alpha.ndim == 2:
            alpha = alpha[:, :, np.newaxis]
        blended = img1.astype(np.float32) * (1 - alpha) + img2.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error blending images: {e}")
        sys.exit()

# ==== Deep Dream Functions ====
def preprocess_image_for_deep_dream(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return image_tensor

def postprocess_deep_dream_output(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    tensor = tensor.clamp(0,1)
    image = transforms.ToPILImage()(tensor)
    return np.array(image)

def load_deep_dream_model():
    print("Loading InceptionV3 model for Deep Dream...")
    try:
        model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT,
                                    aux_logits=True).to(DEVICE).eval()
        print("InceptionV3 model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading InceptionV3 model: {e}")
        sys.exit()

def deep_dream_processing(image, model, layers, iterations=20, lr=0.02):
    try:
        original_size = image.shape[:2][::-1]  # (width, height)
        min_size = 224
        pil_img = Image.fromarray(image)
        if pil_img.width < min_size or pil_img.height < min_size:
            pil_img = pil_img.resize((max(pil_img.width, min_size), max(pil_img.height, min_size)), Image.LANCZOS)
            resized_for_processing = True
        else:
            resized_for_processing = False

        input_tensor = preprocess_image_for_deep_dream(pil_img)
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)
        hooks = []
        for name, module in model.named_modules():
            if name in layers:
                hooks.append(module.register_forward_hook(hook_fn))
        if not hooks:
            print("Warning: No hooks registered. Check layer names.")
        input_tensor.requires_grad_(True)
        optimizer = torch.optim.Adam([input_tensor], lr=lr, weight_decay=1e-4)
        for i in range(iterations):
            optimizer.zero_grad()
            activations.clear()
            output = model(input_tensor)
            if isinstance(output, tuple):
                _ = output[0]
            if activations:
                loss = torch.stack([act.norm() for act in activations]).sum()
                loss.backward()
                optimizer.step()
                if (i+1) % 5 == 0:
                    print(f"Deep Dream Iteration {i+1}/{iterations}, Loss: {loss.item():.4f}")
            else:
                print("No activations collected; skipping optimization.")
                break
        for hook in hooks:
            hook.remove()
        output_image = postprocess_deep_dream_output(input_tensor)
        if resized_for_processing:
            output_image = cv2.resize(output_image, original_size, interpolation=cv2.INTER_LINEAR)
        return output_image
    except Exception as e:
        print(f"Error during deep dream processing: {e}")
        return image

def print_model_layers(model):
    print("Model layers:")
    for name, module in model.named_modules():
        print(name)

# ==== Object Detection and Image Generation Functions ====
def load_yolo_model():
    print("Loading YOLOv5 model for object detection...")
    try:
        pil_image = Image.fromarray(np.zeros((512,512,3),dtype=np.uint8))
        height, width = pil_image.size
        print(f"Image for detection - Size: {pil_image.size}")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True).to(DEVICE).eval()
        model.conf = 0.15
        model.iou = 0.3
        print("YOLOv5 model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        sys.exit()

# --- UPDATED EfficientDet loader for inference ---
def load_efficientdet_model():
    print("Loading EfficientDet model for object detection...")
    try:
        from effdet import create_model, DetBenchPredict
        model = create_model('tf_efficientdet_d0', pretrained=True)
        # Wrap the model with DetBenchPredict so that it returns post-processed predictions.
        model = DetBenchPredict(model)
        model = model.to(DEVICE)
        model.eval()
        print("EfficientDet model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading EfficientDet model: {e}")
        sys.exit(1)

def detect_objects_yolo(image, model):
    try:
        pil_image = Image.fromarray(image)
        height, width = pil_image.size
        print(f"Image for detection - Size: {pil_image.size}")
        model.conf = 0.15  # More sensitive detection
        model.iou = 0.3    # Allow more overlapping detections
        results = model(pil_image, size=500)
        detected_objects = results.pandas().xyxy[0]
        if not detected_objects.empty:
            detected_objects['width'] = detected_objects['xmax'] - detected_objects['xmin']
            detected_objects['height'] = detected_objects['ymax'] - detected_objects['ymin']
            detected_objects['area_ratio'] = ((detected_objects['width'] * detected_objects['height']) /
                                              (width * height))
            max_area = width * height * MAX_OBJECT_SIZE_RATIO
            detected_objects = detected_objects[(
                detected_objects['width'] * detected_objects['height']) < max_area]
            detected_objects = detected_objects.nlargest(MAX_OBJECTS_PER_FRAME, 'confidence')
            if not detected_objects.empty:
                print(f"Detected {len(detected_objects)} valid objects (YOLO): {', '.join(detected_objects['name'].unique())}")
                print(f"Confidence scores: {', '.join(f'{c:.2f}' for c in detected_objects['confidence'])}")
        return detected_objects
    except Exception as e:
        print(f"Error during YOLO object detection: {e}")
        return None

def detect_objects_efficientdet(image, model, min_size=20):
    try:
        from torchvision import transforms
        pil_image = Image.fromarray(image)
        original_width, original_height = pil_image.size
        target_size = 512
        pil_image_resized = pil_image.resize((target_size, target_size), Image.LANCZOS)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(pil_image_resized).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw_outputs = model(img_tensor)
        # Debug logging
        print("\nDEBUG: Model Output Analysis")
        print(f"Type of raw_outputs: {type(raw_outputs)}")
        if isinstance(raw_outputs, (list, tuple)):
            print(f"Length of raw_outputs: {len(raw_outputs)}")
            if len(raw_outputs) > 0:
                print(f"Type of first element: {type(raw_outputs[0])}")
        elif isinstance(raw_outputs, dict):
            print(f"Keys in raw_outputs dict: {list(raw_outputs.keys())}")
        elif torch.is_tensor(raw_outputs):
            print(f"Shape of raw_outputs tensor: {raw_outputs.shape}")
        outputs = None
        # Handle direct tensor output of shape [batch, num_detections, 6]
        if torch.is_tensor(raw_outputs) and raw_outputs.ndim == 3 and raw_outputs.shape[-1] >= 6:
            # Adjust confidence threshold based on creativity
            adjusted_threshold = BASE_CONFIDENCE_THRESHOLD * (1.0 - CREATIVITY * 0.7)
            scores = raw_outputs[..., 4]
            mask = scores > adjusted_threshold
            if mask.any():
                outputs = {
                    "boxes": raw_outputs[..., :4][mask],
                    "scores": scores[mask],
                    "labels": raw_outputs[..., 5][mask]
                }
        # Handle dictionary output
        elif isinstance(raw_outputs, dict):
            if "detections" in raw_outputs:
                detections = raw_outputs["detections"]
                if torch.is_tensor(detections) and detections.shape[-1] >= 6:
                    outputs = {
                        "boxes": detections[..., :4],
                        "scores": detections[..., 4],
                        "labels": detections[..., 5]
                    }
            else:
                required_keys = {"boxes", "scores", "labels"}
                if all(key in raw_outputs for key in required_keys):
                    outputs = raw_outputs
        if outputs is None:
            error_msg = "Could not interpret model outputs in a supported format\n"
            error_msg += "Expected format options:\n"
            error_msg += "1. Dict with 'detections' key containing tensor of shape [N, 6+]\n"
            error_msg += "2. Dict with 'boxes', 'scores', and 'labels' keys\n"
            error_msg += "3. List/Tuple where first element is tensor of shape [N, 6+]\n"
            error_msg += "4. List/Tuple where first element is dict with required keys"
            raise ValueError(error_msg)
        def convert_output(x):
            if torch.is_tensor(x):
                return x.cpu().numpy()
            elif isinstance(x, list):
                try:
                    return np.array(x)
                except Exception:
                    return np.vstack(x)
            else:
                return np.array(x)
        try:
            boxes = convert_output(outputs["boxes"])
            scores = convert_output(outputs["scores"])
            labels = convert_output(outputs["labels"])
        except (KeyError, TypeError) as e:
            raise ValueError(f"Failed to extract detection components from outputs: {e}")
        if boxes.ndim != 2:
            boxes = boxes.reshape(-1, 4)
        scale_x = original_width / target_size
        scale_y = original_height / target_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        import pandas as pd
        data = []
        score_thresh = 0.15
        for i, box in enumerate(boxes):
            if scores[i] < score_thresh:
                continue
            x_min, y_min, x_max, y_max = box
            if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
                continue
            data.append({
                'xmin': x_min,
                'ymin': y_min,
                'xmax': x_max,
                'ymax': y_max,
                'confidence': scores[i],
                'name': str(labels[i])
            })
        df = pd.DataFrame(data)
        if not df.empty:
            print(f"Detected {len(df)} valid objects (EfficientDet): {', '.join(df['name'].unique())}")
        return df
    except Exception as e:
        print(f"Error during EfficientDet object detection: {e}")
        return None

def detect_objects_in_patches(image, model, detect_fn, min_size=20):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    detections = detect_fn(image, model)
    if detections is not None and not detections.empty:
        detections['width'] = detections['xmax'] - detections['xmin']
        detections['height'] = detections['ymax'] - detections['ymin']
        detections = detections[((detections['width'] >= min_size) & 
                                 (detections['height'] >= min_size))]
        for _, obj in detections.iterrows():
            draw.rectangle(
                [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']], 
                outline="red", 
                width=2
            )
            draw.text(
                (obj['xmin'], obj['ymin'] - 10),
                f"{obj['name']} {obj['confidence']:.2f}",
                fill="red"
            )
        print(f"Found {len(detections)} valid objects: {', '.join(detections['name'].unique())}")
        pil_image.save("detections_debug.jpg")
        print("Saved detections_debug.jpg with bounding boxes and labels.")
        return detections
    print("No objects detected.")
    return pd.DataFrame()

def load_stable_diffusion_pipeline():
    """Load the Stable Diffusion pipeline for image generation."""
    print("Loading Stable Diffusion pipeline for image generation...")
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32
        ).to(DEVICE)
        print("Stable Diffusion pipeline loaded successfully!")
        return pipeline
    except Exception as e:
        print(f"Error loading Stable Diffusion pipeline: {e}")
        sys.exit()

def generate_image(obj_name, pipeline):
    try:
        style = random.choice(ARTISTIC_STYLES)
        prompt_template = random.choice(style['prompts'])
        prompt = prompt_template.format(obj=obj_name)
        
        with torch.autocast(DEVICE):
            generated_image = pipeline(
                prompt, 
                guidance_scale=style['guidance_scale']
            ).images[0]
        print(f"Generated {style['name']} image for {obj_name} with prompt: '{prompt}'")
        return generated_image
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

def morph_detected_objects(current_frame, detected_objects, pipeline, deep_dream_model, dream_layers, deep_dream_iterations, deep_dream_lr, morph_states, frame_counter):
    try:
        pil_frame = Image.fromarray(current_frame)
        active_morphs = []
        for morph in list(morph_states):
            if morph.update():
                region = pil_frame.crop(morph.region)
                blend_factor = morph.get_blend_factor()
                mask = Image.new('L', region.size, int(255 * blend_factor))
                mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
                blended_region = Image.composite(morph.generated_image, region, mask)
                pil_frame.paste(blended_region, (morph.region[0], morph.region[1]))
                active_morphs.append(morph)
            else:
                print(f"Completed morphing for object {morph.obj_id}")
        if len(detected_objects) > 0:
            print(f"Starting morph for {len(detected_objects)} object(s)...")
            for index, obj in detected_objects.iterrows():
                class_name = obj['name']
                x_min, y_min, x_max, y_max = map(int, [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
                obj_id = f"{class_name}_{index}_{frame_counter}"
                print(f"Processing object: {class_name} at [{x_min}, {y_min}, {x_max}, {y_max}]")
                overlap = False
                for morph in active_morphs:
                    old_x1, old_y1, old_x2, old_y2 = morph.region
                    if (x_min < old_x2 and x_max > old_x1 and 
                        y_min < old_y2 and y_max > old_y1):
                        overlap = True
                        break
                if not overlap:
                    generated_image = generate_image(class_name, pipeline)
                    if generated_image is not None:
                        region = pil_frame.crop((x_min, y_min, x_max, y_max))
                        generated_resized = generated_image.resize(region.size)
                        generated_dreamed = deep_dream_processing(
                            np.array(generated_resized),
                            model=deep_dream_model,
                            layers=dream_layers,
                            iterations=deep_dream_iterations,
                            lr=deep_dream_lr
                        )
                        generated_dreamed_image = Image.fromarray(generated_dreamed)
                        morph_state = MorphState(
                            obj_id,
                            (x_min, y_min, x_max, y_max),
                            generated_dreamed_image,
                            frame_counter
                        )
                        active_morphs.append(morph_state)
                    else:
                        print(f"Failed to generate image for {class_name}. Skipping morphing.")
        morph_states.clear()
        morph_states.extend(active_morphs)
        return np.array(pil_frame)
    except Exception as e:
        print(f"Error during morphing: {e}")
        return current_frame

def zoom_image(image, zoom_factor):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    zoom_width = int(width / zoom_factor)
    zoom_height = int(height / zoom_factor)
    x1 = max(0, center_x - zoom_width // 2)
    y1 = max(0, center_y - zoom_height // 2)
    x2 = min(width, x1 + zoom_width)
    y2 = min(height, y1 + zoom_height)
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

class MorphState:
    def __init__(self, obj_id, region, generated_image, frame_start):
        self.obj_id = obj_id
        self.region = region
        self.generated_image = generated_image
        self.frame_start = frame_start
        self.frame_count = 0
    def get_blend_factor(self):
        progress = min(1.0, self.frame_count / MORPH_FRAMES)
        blend = MORPH_MIN_BLEND + (MORPH_MAX_BLEND - MORPH_MIN_BLEND) * progress
        return blend
    def update(self):
        self.frame_count += 1
        return self.frame_count < MORPH_FRAMES

# ==== Main Function ====
def main(input_path, output_path=None, deep_dream_iterations=30, deep_dream_lr=0.02, creativity=0.5):
    if output_path is None:
        input_filename = os.path.basename(input_path)
        base_name = os.path.splitext(input_filename)[0]
        output_path = os.path.join('keep', f'{base_name}_dream.mp4')
    else:
        output_path = os.path.join('keep', output_path)
    os.makedirs('keep', exist_ok=True)
    global DEFAULT_DIVISOR, CREATIVITY
    grid_scale = DEFAULT_GRID_SCALE
    divisor = DEFAULT_DIVISOR
    blend_alpha = DEFAULT_BLEND_ALPHA
    zoom_enabled = False
    current_zoom = MIN_ZOOM
    object_detection_enabled = True
    CREATIVITY = creativity  # Initialize from parameter

    content_image_array = None
    try:
        cap = None
        if is_video_file(input_path):
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file {input_path}")
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video opened: {frame_count} frames at {input_fps} FPS")
            ret, frame = cap.read()
            if not ret:
                raise Exception("Cannot read the first frame of the video.")
            content_image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            content_image_array = np.array(Image.open(input_path).convert('RGB'))
            input_fps = FPS
        if content_image_array is None:
            raise Exception(f"Failed to load content from {input_path}")
        original_height, original_width, _ = content_image_array.shape
        max_dimension = 800
        scaling_factor = max_dimension / max(original_height, original_width)
        if max(original_height, original_width) > max_dimension:
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            content_image_array = cv2.resize(content_image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = content_image_array.shape[:2]
            print(f"Resized image to {width}x{height} for performance")
        else:
            height, width = original_height, original_width
            print(f"Image size is {width}x{height}; no resizing needed")
        pad_height = (grid_scale - height % grid_scale) if height % grid_scale != 0 else 0
        pad_width = (grid_scale - width % grid_scale) if width % grid_scale != 0 else 0
        if pad_height or pad_width:
            content_image_array = cv2.copyMakeBorder(content_image_array, 0, pad_height, 0, pad_width, cv2.BORDER_REPLICATE)
            height, width = content_image_array.shape[:2]
            print(f"Padded image to {width}x{height} for GRID_SCALE={grid_scale}")
        else:
            print("No padding needed")
        cumulative_dream_image_array = content_image_array.copy()
        deep_dream_model = load_deep_dream_model()
        yolo_model = load_yolo_model()
        efficientdet_model = load_efficientdet_model()
        sd_pipeline = load_stable_diffusion_pipeline()
        pygame.init()
        overlay_font = pygame.font.SysFont("Arial", 14)
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()
        print(f"Pygame window initialized with size {width}x{height}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = input_fps if cap is not None else FPS
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Video writer initialized: '{output_path}'")
        grid_shape = (height // grid_scale, width // grid_scale)
        grid = init_game_of_life_grid(grid_shape)
        rules = ['Conway', 'HighLife', 'Seeds', 'Custom', 'DivisorRule']
        rule_index = 0
        current_rule = rules[rule_index]
        pygame.display.set_caption(f"Cellular Automata Rule: {current_rule}")
        print(f"Initial cellular automata rule: {current_rule}")
        previous_output = None
        frame_counter = 0
        use_yolo = True  # Toggle between YOLOv5 and EfficientDet
        running = True
        while running:
            frame_counter += 1
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
                        print(f"Switched rule to: {current_rule}")
                    elif event.key == pygame.K_r:
                        grid = init_game_of_life_grid(grid_shape)
                        cumulative_dream_image_array = content_image_array.copy()
                        previous_output = None
                        current_zoom = MIN_ZOOM
                        print("Reset grid, image, and zoom.")
                    elif event.key == pygame.K_LEFT:
                        grid_scale = max(4, grid_scale - 4)
                        grid_shape = (height // grid_scale, width // grid_scale)
                        grid = init_game_of_life_grid(grid_shape)
                        print(f"Decreased grid scale to {grid_scale}")
                    elif event.key == pygame.K_RIGHT:
                        grid_scale += 4
                        grid_shape = (height // grid_scale, width // grid_scale)
                        grid = init_game_of_life_grid(grid_shape)
                        print(f"Increased grid scale to {grid_scale}")
                    elif event.key == pygame.K_UP:
                        divisor += 1
                        print(f"Increased divisor to {divisor}")
                    elif event.key == pygame.K_DOWN:
                        divisor = max(1, divisor - 1)
                        print(f"Decreased divisor to {divisor}")
                    elif event.key == pygame.K_b:
                        blend_alpha = min(1.0, blend_alpha + 0.1)
                        print(f"Increased blend alpha to {blend_alpha:.1f}")
                    elif event.key == pygame.K_n:
                        blend_alpha = max(0.0, blend_alpha - 0.1)
                        print(f"Decreased blend alpha to {blend_alpha:.1f}")
                    elif event.key == pygame.K_o:
                        object_detection_enabled = not object_detection_enabled
                        print(f"Object detection/morphing {'enabled' if object_detection_enabled else 'disabled'}")
                    elif event.key == pygame.K_z:
                        zoom_enabled = not zoom_enabled
                        if not zoom_enabled:
                            current_zoom = MIN_ZOOM
                        print(f"Zoom {'enabled' if zoom_enabled else 'disabled'}")
                    elif event.key == pygame.K_LEFTBRACKET:
                        CREATIVITY = max(0.0, CREATIVITY - 0.1)
                        print(f"Decreased creativity to {CREATIVITY:.1f}")
                    elif event.key == pygame.K_RIGHTBRACKET:
                        CREATIVITY = min(1.0, CREATIVITY + 0.1)
                        print(f"Increased creativity to {CREATIVITY:.1f}")
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached.")
                    running = False
                    break
                content_image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                content_image_array = cv2.resize(content_image_array, (width, height))
            if previous_output is not None:
                content_image_array = cv2.addWeighted(content_image_array, 0.5, previous_output, 0.5, 0)
            grid = update_grid(grid, current_rule, divisor)
            if np.sum(grid) == 0:
                grid = init_game_of_life_grid(grid_shape)
                print("Grid died out; reinitialized.")
            upscaled_grid = cv2.resize(grid.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            alive_mask = upscaled_grid
            dead_mask = 1 - upscaled_grid
            composite_image = (
                (content_image_array.astype(np.float32) * blend_alpha +
                 cumulative_dream_image_array.astype(np.float32) * (1 - blend_alpha)) * alive_mask[:, :, np.newaxis] +
                cumulative_dream_image_array.astype(np.float32) * dead_mask[:, :, np.newaxis]
            ).astype(np.uint8)
            deep_dreamed_image_array = deep_dream_processing(
                composite_image,
                model=deep_dream_model,
                layers=DREAM_LAYERS,
                iterations=deep_dream_iterations,
                lr=deep_dream_lr
            )
            cumulative_dream_image_array = deep_dreamed_image_array.copy()
            if zoom_enabled:
                current_zoom += ZOOM_SPEED
                if current_zoom > MAX_ZOOM:
                    current_zoom = MIN_ZOOM
            cumulative_dream_image_array = zoom_image(cumulative_dream_image_array, current_zoom)
            if not hasattr(main, 'morph_states'):
                main.morph_states = []
            if object_detection_enabled and frame_counter % OBJECT_DETECTION_INTERVAL == 0:
                print("Performing object detection and morphing...")
                if use_yolo:
                    detected_objects = detect_objects_in_patches(cumulative_dream_image_array, yolo_model, detect_objects_yolo)
                    print("Using YOLOv5 for detection.")
                else:
                    detected_objects = detect_objects_in_patches(cumulative_dream_image_array, efficientdet_model, detect_objects_efficientdet)
                    print("Using EfficientDet for detection.")
                use_yolo = not use_yolo
                if detected_objects is not None and not detected_objects.empty:
                    cumulative_dream_image_array = morph_detected_objects(
                        cumulative_dream_image_array,
                        detected_objects,
                        sd_pipeline,
                        deep_dream_model,
                        DREAM_LAYERS,
                        deep_dream_iterations,
                        deep_dream_lr,
                        main.morph_states,
                        frame_counter
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    print("No objects detected in this iteration.")
            elif len(main.morph_states) > 0:
                cumulative_dream_image_array = morph_detected_objects(
                    cumulative_dream_image_array,
                    pd.DataFrame(),
                    sd_pipeline,
                    deep_dream_model,
                    DREAM_LAYERS,
                    deep_dream_iterations,
                    deep_dream_lr,
                    main.morph_states,
                    frame_counter
                )
            previous_output = content_image_array.copy()
            display_image = cumulative_dream_image_array
            try:
                surface = pygame.surfarray.make_surface(np.swapaxes(display_image, 0, 1))
            except Exception as e:
                print(f"Error creating Pygame surface: {e}")
                sys.exit()
            overlay_texts = [
                f"SPACE - Rule: {current_rule}",
                f"←/→ - Grid Scale: {grid_scale}",
                f"↑/↓ - Divisor: {divisor}",
                f"B/N - Blend: {blend_alpha:.1f}",
                f"[/] - Creativity: {CREATIVITY:.1f}",
                f"Z - Zoom: {'On' if zoom_enabled else 'Off'}",
                f"O - Objects: {'On' if object_detection_enabled else 'Off'}",
                "R - Reset | ESC - Quit"
            ]
            y_offset = 5
            for line in overlay_texts:
                text_surface = overlay_font.render(line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(topleft=(5, y_offset))
                bg_surface = pygame.Surface((text_rect.width, text_rect.height))
                bg_surface.set_alpha(150)
                bg_surface.fill((0, 0, 0))
                surface.blit(bg_surface, text_rect)
                surface.blit(text_surface, text_rect)
                y_offset += text_rect.height + 2
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            try:
                output_frame = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
                video_writer.write(output_frame)
            except Exception as e:
                print(f"Error writing frame to video: {e}")
                sys.exit()
            clock.tick(FPS)
        if cap is not None:
            cap.release()
        video_writer.release()
        pygame.quit()
        print(f"Video saved as '{output_path}'")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cellular Automata with Deep Dream, Zoom, and Object Morphing.')
    parser.add_argument('input_file', type=str, help='Path to the content image or video file.')
    parser.add_argument('--output_file', type=str, help='Path to the output video file (optional, defaults to input_name_dream.mp4 in keep/')
    parser.add_argument('--creativity', type=float, default=0.5, help='Creativity level (0.0 to 1.0) - higher values allow more flexible object matching')
    parser.add_argument('--deep_dream_iterations', type=int, default=30, help='Number of Deep Dream iterations.')
    parser.add_argument('--deep_dream_lr', type=float, default=0.02, help='Learning rate for Deep Dream.')
    args = parser.parse_args()

    main(
        input_path=args.input_file,
        output_path=args.output_file,
        deep_dream_iterations=args.deep_dream_iterations,
        creativity=args.creativity,
        deep_dream_lr=args.deep_dream_lr
    )
