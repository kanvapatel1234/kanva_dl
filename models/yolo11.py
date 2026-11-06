"""
YOLO11 Model Implementation for Indoor Object Detection.

This module loads a pretrained YOLO11 model using the Ultralytics framework,
performs object detection on an input image, extracts labels, confidence scores,
and bounding boxes, and generates an annotated output image.
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
from ultralytics import YOLO

# Module-level model cache to ensure the model is loaded only once and reused
_CACHED_MODEL = None
_CACHED_DEVICE = None

# Default confidence threshold for YOLO11
DEFAULT_CONF_THRESHOLD = 0.35

# Output directory for saving annotated images
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))

# Path to local weights file or default model name
WEIGHTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolo11n.pt"))


def get_device() -> str:
    """Detect and return the appropriate execution device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_class_color(class_name: str) -> tuple:
    """Generate a consistent, visually distinct BGR color for a given class name."""
    hash_val = sum(ord(c) for c in class_name) * 57 + len(class_name) * 31
    hue = (hash_val % 180)
    hsv_pixel = np.uint8([[[hue, 210, 240]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


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


def load_model(device: Optional[str] = None):
    """
    Load and cache pretrained YOLO11 model weights.

    Args:
        device: Device string ('cuda' or 'cpu'). Defaults to auto-detected device.

    Returns:
        Tuple of (model, device).
    """
    global _CACHED_MODEL, _CACHED_DEVICE

    if device is None:
        device = get_device()

    if _CACHED_MODEL is not None and _CACHED_DEVICE == device:
        return _CACHED_MODEL, _CACHED_DEVICE

    weights_path = WEIGHTS_FILE if os.path.exists(WEIGHTS_FILE) else "yolo11n.pt"
    print(f"[YOLO11] Loading pretrained YOLO11 ({weights_path}) on {device}...")

    model = YOLO(weights_path)

    _CACHED_MODEL = model
    _CACHED_DEVICE = device
    print("[YOLO11] Pretrained model loaded successfully.")

    return _CACHED_MODEL, _CACHED_DEVICE


def run_model(
    image_input: Union[str, Image.Image],
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    device: Optional[str] = None,
    save_to_disk: bool = True
) -> Dict[str, Any]:
    """
    Execute YOLO11 detection on the specified image.

    Args:
        image_input: File path string or PIL Image object.
        conf_threshold: Minimum confidence score to retain detections.
        device: Execution device ('cpu' or 'cuda'). Defaults to auto-detect.
        save_to_disk: Whether to save annotated image to outputs/ directory.

    Returns:
        Structured dictionary with detection results:
        {
            "model_name": "YOLO11",
            "device": str,
            "inference_time_ms": float,
            "detections": [
                {"label": str, "confidence": float, "box": [x1, y1, x2, y2]}
            ],
            "annotated_image": np.ndarray (RGB),
            "annotated_image_base64": str,
            "annotated_image_path": str
        }
    """
    # Load model and device
    model, exec_device = load_model(device=device)

    # Load original image for custom uniform annotation
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Input image not found: {image_input}")
        pil_image = Image.open(image_input).convert("RGB")
        source = image_input
    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
        source = pil_image
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    image_np = np.array(pil_image)

    # Run inference and measure latency
    if exec_device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    results = model.predict(
        source=source,
        conf=conf_threshold,
        device=exec_device,
        verbose=False
    )

    if exec_device == "cuda":
        torch.cuda.synchronize()
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    result = results[0]
    names = result.names

    detections: List[Dict[str, Any]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_idx in zip(boxes_xyxy, confs, classes):
            class_name = names.get(cls_idx, str(cls_idx))
            detections.append({
                "label": str(class_name),
                "confidence": float(conf),
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            })

    # Generate annotated image
    annotated_rgb = annotate_image(image_np, detections)
    b64_str = image_to_base64(annotated_rgb)

    output_path = ""
    if save_to_disk:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "yolo11_annotated.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

    return {
        "model_name": "YOLO11",
        "device": str(exec_device),
        "inference_time_ms": round(inference_time_ms, 2),
        "detections": detections,
        "annotated_image": annotated_rgb,
        "annotated_image_base64": b64_str,
        "annotated_image_path": output_path
    }


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else "sample_indoor_living_room.jpg"
    print(f"Testing YOLO11 on {test_img}...")
    res = run_model(test_img)
    print(f"Detected {len(res['detections'])} objects in {res['inference_time_ms']:.2f} ms.")
    for d in res["detections"][:5]:
        print(f" - {d['label']}: {d['confidence']:.3f} at {d['box']}")
