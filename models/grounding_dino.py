"""
Grounding DINO Model Implementation for Indoor Object Detection.

This module loads a pretrained Grounding DINO model from Hugging Face Transformers,
accepts configurable open-vocabulary text prompts for indoor objects, performs zero-shot
object detection on an input image, extracts labels, confidence scores, and bounding boxes,
and generates an annotated output image.
"""

import os
import io
import time
import base64
from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image
import torch
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Module-level model cache to ensure the model is loaded only once and reused
_CACHED_MODEL = None
_CACHED_PROCESSOR = None
_CACHED_DEVICE = None

# Default indoor object categories as requested
DEFAULT_INDOOR_CLASSES = [
    "bed", "chair", "table", "laptop", "keyboard", "mouse",
    "bottle", "cup", "book", "television", "monitor", "clock",
    "plant", "sofa", "person"
]

# Model identifier
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Default confidence thresholds for Grounding DINO
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.25

# Output directory for saving annotated images
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def get_device() -> torch.device:
    """Detect and return the appropriate PyTorch device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_color(class_name: str) -> tuple:
    """Generate a consistent, visually distinct BGR color for a given class name."""
    hash_val = sum(ord(c) for c in class_name) * 57 + len(class_name) * 31
    hue = (hash_val % 180)
    hsv_pixel = np.uint8([[[hue, 210, 240]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def format_prompt(classes: Union[List[str], str]) -> str:
    """
    Format a list of object classes into Grounding DINO's required dot-separated syntax.
    Example: 'bed . chair . table . laptop .'
    """
    if isinstance(classes, str):
        text = classes.strip().lower()
        if not text.endswith("."):
            text += " ."
        return text

    cleaned = [c.strip().lower() for c in classes if c.strip()]
    return " . ".join(cleaned) + " ."


def annotate_image(
    image_rgb: np.ndarray,
    detections: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Annotate an RGB image with bounding boxes, class labels, and confidence scores.

    Args:
        image_rgb: RGB image as numpy uint8 array.
        detections: List of detection dictionaries containing 'label', 'confidence', 'box'.

    Returns:
        Annotated RGB image as numpy uint8 array.
    """
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = det["label"]
        conf = det["confidence"]
        color = get_class_color(label)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

        # Prepare label banner
        caption = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(caption, font, font_scale, thickness)

        # Position banner above the bounding box if possible, else inside
        banner_y1 = max(0, y1 - th - baseline - 4)
        banner_y2 = y1 if y1 - th - baseline - 4 >= 0 else y1 + th + baseline + 4
        banner_x2 = min(annotated.shape[1], x1 + tw + 6)

        # Draw banner background and text
        cv2.rectangle(annotated, (x1, banner_y1), (banner_x2, banner_y2), color, -1)
        text_y = y1 - 4 if y1 - th - baseline - 4 >= 0 else y1 + th + 2
        cv2.putText(
            annotated,
            caption,
            (x1 + 3, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def image_to_base64(img_rgb: np.ndarray, format: str = "JPEG") -> str:
    """Convert RGB numpy array to data URL base64 string."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=92)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"


def load_model(device: Optional[torch.device] = None):
    """
    Load and cache pretrained Grounding DINO model and processor.

    Args:
        device: PyTorch device to load model onto. Defaults to auto-detected device.

    Returns:
        Tuple of (model, processor, device).
    """
    global _CACHED_MODEL, _CACHED_PROCESSOR, _CACHED_DEVICE

    if device is None:
        device = get_device()

    if _CACHED_MODEL is not None and _CACHED_PROCESSOR is not None and _CACHED_DEVICE == device:
        return _CACHED_MODEL, _CACHED_PROCESSOR, _CACHED_DEVICE

    print(f"[Grounding DINO] Loading pretrained Grounding DINO ({MODEL_ID}) on {device}...")

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)

    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_PROCESSOR = processor
    _CACHED_DEVICE = device
    print("[Grounding DINO] Pretrained model loaded successfully.")

    return _CACHED_MODEL, _CACHED_PROCESSOR, _CACHED_DEVICE


def run_model(
    image_input: Union[str, Image.Image],
    conf_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    prompt: Optional[Union[List[str], str]] = None,
    device: Optional[torch.device] = None,
    save_to_disk: bool = True
) -> Dict[str, Any]:
    """
    Execute Grounding DINO zero-shot detection on the specified image.

    Args:
        image_input: File path string or PIL Image object.
        conf_threshold: Box confidence threshold.
        text_threshold: Text alignment threshold.
        prompt: List of class strings or formatted dot-separated prompt string.
                Defaults to DEFAULT_INDOOR_CLASSES.
        device: PyTorch device (CPU or CUDA). Defaults to auto-detect.
        save_to_disk: Whether to save annotated image to outputs/ directory.

    Returns:
        Structured dictionary with detection results:
        {
            "model_name": "Grounding DINO",
            "device": str,
            "inference_time_ms": float,
            "detections": [
                {"label": str, "confidence": float, "box": [x1, y1, x2, y2]}
            ],
            "annotated_image": np.ndarray (RGB),
            "annotated_image_base64": str,
            "annotated_image_path": str,
            "prompt_used": str
        }
    """
    # Load model, processor, and device
    model, processor, exec_device = load_model(device=device)

    # Prepare indoor text prompt
    classes_to_use = prompt if prompt is not None else DEFAULT_INDOOR_CLASSES
    formatted_prompt = format_prompt(classes_to_use)

    # Load and preprocess image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Input image not found: {image_input}")
        pil_image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    image_np = np.array(pil_image)
    image_w, image_h = pil_image.size

    # Tokenize text and prepare model inputs
    inputs = processor(
        images=pil_image,
        text=formatted_prompt,
        return_tensors="pt"
    ).to(exec_device)

    # Perform inference and record latency
    if exec_device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)

    if exec_device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    # Post-process detections
    post_results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        threshold=conf_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image_h, image_w)]
    )[0]

    raw_labels = post_results.get("text_labels", post_results.get("labels", []))
    raw_scores = post_results["scores"].detach().cpu().numpy()
    raw_boxes = post_results["boxes"].detach().cpu().numpy()

    detections: List[Dict[str, Any]] = []
    for label_item, score, box in zip(raw_labels, raw_scores, raw_boxes):
        label_str = str(label_item).strip().lower()
        if not label_str:
            continue

        x1 = max(0.0, min(float(box[0]), float(image_w)))
        y1 = max(0.0, min(float(box[1]), float(image_h)))
        x2 = max(0.0, min(float(box[2]), float(image_w)))
        y2 = max(0.0, min(float(box[3]), float(image_h)))

        if x2 > x1 and y2 > y1:
            detections.append({
                "label": label_str,
                "confidence": float(score),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

    # Generate annotated image
    annotated_rgb = annotate_image(image_np, detections)
    b64_str = image_to_base64(annotated_rgb)

    output_path = ""
    if save_to_disk:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "grounding_dino_annotated.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

    return {
        "model_name": "Grounding DINO",
        "device": str(exec_device),
        "inference_time_ms": round(inference_time_ms, 2),
        "detections": detections,
        "annotated_image": annotated_rgb,
        "annotated_image_base64": b64_str,
        "annotated_image_path": output_path,
        "prompt_used": formatted_prompt
    }


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else "sample_indoor_living_room.jpg"
    print(f"Testing Grounding DINO on {test_img}...")
    res = run_model(test_img)
    print(f"Detected {len(res['detections'])} objects in {res['inference_time_ms']:.2f} ms.")
    for d in res["detections"][:5]:
        print(f" - {d['label']}: {d['confidence']:.3f} at {d['box']}")
