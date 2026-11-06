"""
FastAPI Interactive Web Application for Indoor Object Detection 4-Model Comparative Analysis.

Provides:
- Web frontend (HTML/CSS/JS) for selecting/uploading images, tweaking prompts, and viewing
  complete interactive outputs directly in the browser (no terminal or outputs folder browsing needed).
- REST API for hardware status, sample image loading, and real-time comparative inference across:
  1. Faster R-CNN (Two-stage CNN)
  2. YOLO11 (One-stage CNN)
  3. RT-DETR (Real-time Detection Transformer)
  4. Grounding DINO (Open-vocabulary Zero-shot Transformer)
"""

import os
import io
import base64
import tempfile
import webbrowser
from typing import Optional

# Ensure writable Matplotlib cache directory and thread-safe headless backend
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_dl_aat"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import torch

# Import model runners
from models.faster_rcnn import run_model as run_faster_rcnn
from models.yolo11 import run_model as run_yolo11
from models.rt_detr import run_model as run_rt_detr
from models.grounding_dino import run_model as run_grounding_dino

# Import comparison analysis pipeline
from analysis.comparison import run_comparison

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
INDEX_HTML_PATH = os.path.join(TEMPLATES_DIR, "index.html")
SAMPLE_IMAGE_PATH = os.path.join(BASE_DIR, "sample_indoor_living_room.jpg")

# Initialize FastAPI application
app = FastAPI(
    title="Indoor Object Detection Comparative Analysis",
    description="Comparative analysis of Faster R-CNN, YOLO11, RT-DETR, and Grounding DINO",
    version="2.0.0"
)

# Enable CORS for maximum client compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def pil_to_base64(pil_img: Image.Image, format: str = "JPEG") -> str:
    """Convert a PIL Image to a base64 data URL string."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=92)
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{data}"


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the complete interactive HTML frontend."""
    if not os.path.exists(INDEX_HTML_PATH):
        raise HTTPException(status_code=404, detail="Frontend template not found.")
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/api/device")
def get_device_info():
    """Return device acceleration details (CUDA or CPU)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return {
            "device": "cuda",
            "device_name": gpu_name,
            "gpu_count": torch.cuda.device_count()
        }
    return {
        "device": "cpu",
        "device_name": "CPU (Intel/AMD Multi-threaded)",
        "gpu_count": 0
    }


@app.get("/api/sample-image")
def get_sample_image():
    """Return the base64-encoded indoor sample image for 1-click testing."""
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        raise HTTPException(status_code=404, detail="Sample indoor image not found.")
    pil_img = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    b64_url = pil_to_base64(pil_img)
    return {
        "filename": "sample_indoor_living_room.jpg",
        "image_base64": b64_url
    }


@app.post("/api/analyze")
def analyze_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    conf_threshold: Optional[float] = Form(None)
):
    """
    Execute inference across all four models on the uploaded image and generate
    interactive comparison metrics, tables, and visualization charts.
    """
    try:
        # Read uploaded image bytes synchronously
        image_bytes = file.file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Process prompt for Grounding DINO
    dino_prompt = None
    if prompt and prompt.strip():
        dino_prompt = prompt.strip()

    # Model threshold overrides if supplied
    threshold_kwargs = {}
    if conf_threshold is not None and conf_threshold > 0:
        threshold_kwargs["conf_threshold"] = conf_threshold

    results_dict = {}

    try:
        # 1. Faster R-CNN
        res_frcnn = run_faster_rcnn(pil_image, save_to_disk=False, **threshold_kwargs)
        results_dict["Faster R-CNN"] = res_frcnn

        # 2. YOLO11
        res_yolo = run_yolo11(pil_image, save_to_disk=False, **threshold_kwargs)
        results_dict["YOLO11"] = res_yolo

        # 3. RT-DETR
        res_rtdetr = run_rt_detr(pil_image, save_to_disk=False, **threshold_kwargs)
        results_dict["RT-DETR"] = res_rtdetr

        # 4. Grounding DINO
        dino_kwargs = {"save_to_disk": False}
        if dino_prompt:
            dino_kwargs["prompt"] = dino_prompt
        if conf_threshold is not None and conf_threshold > 0:
            dino_kwargs["conf_threshold"] = conf_threshold
        res_gdino = run_grounding_dino(pil_image, **dino_kwargs)
        results_dict["Grounding DINO"] = res_gdino

        # Run comparative analysis in memory
        comparison_output = run_comparison(results_dict, save_to_disk=False)

        # Prepare original image base64
        original_b64 = pil_to_base64(pil_image)

        # Prepare cleaned models output for JSON response
        clean_models = {}
        for m_name, m_res in results_dict.items():
            clean_models[m_name] = {
                "model_name": m_res.get("model_name", m_name),
                "device": m_res.get("device", "N/A"),
                "inference_time_ms": m_res.get("inference_time_ms", 0.0),
                "detections": m_res.get("detections", []),
                "annotated_image_base64": m_res.get("annotated_image_base64", "")
            }

        return {
            "success": True,
            "original_image_base64": original_b64,
            "models": clean_models,
            "metrics": comparison_output["metrics"],
            "overlap_info": comparison_output["overlap_info"],
            "table_records": comparison_output["table_records"],
            "charts": {
                "grid_image_base64": comparison_output["grid_image_base64"],
                "counts_chart_base64": comparison_output["counts_chart_base64"],
                "latency_chart_base64": comparison_output["latency_chart_base64"],
                "dashboard_base64": comparison_output["dashboard_base64"]
            },
            "observations": comparison_output["observations"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {e}")


def start_server():
    """Start the FastAPI backend with Uvicorn and automatically open the web browser."""
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("=" * 80)
    print(" INDOOR OBJECT DETECTION: 4-MODEL COMPARATIVE ANALYSIS WEB APPLICATION ")
    print("=" * 80)
    print(f"Starting interactive web server at: {url}")
    print("Opening browser automatically...")
    print("Press CTRL+C in this terminal to stop the server.")
    print("=" * 80)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    start_server()
