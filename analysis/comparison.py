"""
Analysis and Comparison Module for Deep Learning Indoor Object Detection Models.

This module compares four object detection architectures:
1. Faster R-CNN (Two-stage CNN)
2. YOLO11 (One-stage CNN anchor-free)
3. RT-DETR (Real-time Detection Transformer)
4. Grounding DINO (Open-vocabulary Zero-shot Transformer)

It computes metrics, creates Pandas comparison tables, generates Matplotlib visualizations
(including a 2x2 grid, count comparisons, latency charts, and comprehensive dashboards),
and generates detailed scientific observations.
"""

import os
import io
import base64
import tempfile
from typing import Dict, Any, List, Set, Tuple, Optional

# Configure Matplotlib writable directory and non-interactive backend
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_dl_aat"))
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

# Output directory for saving comparison artifacts
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))

# Synonym mapping to enable fair semantic comparison between COCO categories and open prompts
CLASS_SYNONYMS = {
    "couch": "sofa",
    "dining table": "table",
    "tv": "television",
    "television monitor": "television",
    "potted plant": "plant",
    "cell phone": "phone"
}


def fig_to_base64(fig: plt.Figure, dpi: int = 180) -> str:
    """Convert Matplotlib figure to a base64 data URL string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def canonical_class_name(name: str) -> str:
    """Normalize class name to a standard canonical form for fair semantic comparison."""
    cleaned = name.strip().lower()
    return CLASS_SYNONYMS.get(cleaned, cleaned)


def compute_metrics(results_dict: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute quantitative metrics and class overlap statistics across all models.

    Args:
        results_dict: Dictionary mapping model names to their run_model outputs.

    Returns:
        Tuple of (model_metrics, overlap_info).
    """
    model_metrics = {}
    model_classes: Dict[str, Set[str]] = {}

    for model_name, res in results_dict.items():
        detections = res.get("detections", [])
        total_objects = len(detections)
        
        # Raw and canonical classes
        raw_classes = [d["label"] for d in detections]
        canonical_classes = [canonical_class_name(c) for c in raw_classes]
        unique_canonical = set(canonical_classes)
        model_classes[model_name] = unique_canonical

        # Confidence statistics
        confidences = [float(d["confidence"]) for d in detections]
        if confidences:
            avg_conf = float(np.mean(confidences))
            min_conf = float(np.min(confidences))
            max_conf = float(np.max(confidences))
            std_conf = float(np.std(confidences))
        else:
            avg_conf = min_conf = max_conf = std_conf = 0.0

        # Class counts breakdown
        class_distribution = {}
        for c in canonical_classes:
            class_distribution[c] = class_distribution.get(c, 0) + 1

        model_metrics[model_name] = {
            "device": res.get("device", "N/A"),
            "inference_time_ms": float(res.get("inference_time_ms", 0.0)),
            "total_objects": total_objects,
            "unique_classes_count": len(unique_canonical),
            "unique_classes": sorted(list(unique_canonical)),
            "avg_confidence": round(avg_conf, 3),
            "min_confidence": round(min_conf, 3),
            "max_confidence": round(max_conf, 3),
            "std_confidence": round(std_conf, 3),
            "class_distribution": class_distribution,
            "confidences": confidences
        }

    # Cross-model class overlap analysis
    all_models = list(results_dict.keys())
    
    # 1. Objects detected by all four models (Consensus)
    if model_classes:
        consensus_classes = set.intersection(*model_classes.values())
    else:
        consensus_classes = set()

    # 2. Objects missed by individual models (detected by other models but missed by this model)
    missed_by_model: Dict[str, List[str]] = {}
    for m in all_models:
        other_models = [other for other in all_models if other != m]
        other_classes = set().union(*[model_classes[other] for other in other_models]) if other_models else set()
        missed = other_classes - model_classes[m]
        missed_by_model[m] = sorted(list(missed))

    # 3. Objects detected by only one model (Unique discoveries)
    unique_to_model: Dict[str, List[str]] = {}
    for m in all_models:
        other_models = [other for other in all_models if other != m]
        other_classes = set().union(*[model_classes[other] for other in other_models]) if other_models else set()
        unique = model_classes[m] - other_classes
        unique_to_model[m] = sorted(list(unique))

    overlap_info = {
        "consensus_classes": sorted(list(consensus_classes)),
        "missed_by_model": missed_by_model,
        "unique_to_model": unique_to_model
    }

    return model_metrics, overlap_info


def generate_comparison_table(
    model_metrics: Dict[str, Any],
    save_to_disk: bool = True
) -> pd.DataFrame:
    """
    Generate and optionally save a Pandas DataFrame comparing all four models.

    Args:
        model_metrics: Metrics dictionary computed by compute_metrics.
        save_to_disk: Whether to write CSV/MD to outputs folder.

    Returns:
        Pandas DataFrame containing model comparison data.
    """
    rows = []
    for model_name, m in model_metrics.items():
        classes_str = ", ".join(m["unique_classes"]) if m["unique_classes"] else "None"
        rows.append({
            "Model": model_name,
            "Device": m["device"],
            "Inference Time (ms)": m["inference_time_ms"],
            "Total Objects": m["total_objects"],
            "Unique Classes": m["unique_classes_count"],
            "Avg Confidence": m["avg_confidence"],
            "Min Confidence": m["min_confidence"],
            "Max Confidence": m["max_confidence"],
            "Classes Detected": classes_str
        })

    df = pd.DataFrame(rows)

    if save_to_disk:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, "model_comparison_table.csv")
        md_path = os.path.join(OUTPUT_DIR, "model_comparison_table.md")
        df.to_csv(csv_path, index=False)

        headers = list(df.columns)
        md_lines = [
            "| " + " | ".join(str(h) for h in headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        for _, row in df.iterrows():
            md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

    return df


def plot_2x2_grid(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    save_to_disk: bool = True
) -> Tuple[str, str]:
    """
    Display and save the four model annotated images in a clean 2x2 layout.

    Returns:
        Tuple of (saved_file_path, base64_data_url).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    model_names = ["Faster R-CNN", "YOLO11", "RT-DETR", "Grounding DINO"]
    present_models = [m for m in model_names if m in results_dict] + [
        m for m in results_dict if m not in model_names
    ]

    for idx in range(4):
        ax = axes[idx]
        if idx < len(present_models):
            m_name = present_models[idx]
            res = results_dict[m_name]
            img = res.get("annotated_image")
            latency = res.get("inference_time_ms", 0.0)
            det_count = len(res.get("detections", []))

            if img is not None:
                ax.imshow(img)
            ax.set_title(
                f"{m_name}\nDetections: {det_count} | Latency: {latency:.1f} ms",
                fontsize=13,
                fontweight="bold",
                pad=8
            )
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")

        ax.axis("off")

    plt.suptitle(
        "Indoor Object Detection: Pretrained Architecture Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    b64_url = fig_to_base64(fig)
    saved_path = ""
    if save_to_disk:
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, "annotated_comparison_grid_2x2.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        saved_path = output_path

    plt.close(fig)
    return saved_path, b64_url


def plot_object_counts(
    model_metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    save_to_disk: bool = True
) -> Tuple[str, str]:
    """
    Plot bar chart comparing total detections and unique classes across models.

    Returns:
        Tuple of (saved_file_path, base64_data_url).
    """
    models = list(model_metrics.keys())
    total_objects = [model_metrics[m]["total_objects"] for m in models]
    unique_classes = [model_metrics[m]["unique_classes_count"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, total_objects, width, label="Total Objects Detected", color="#2b5c8f")
    rects2 = ax.bar(x + width / 2, unique_classes, width, label="Unique Classes Detected", color="#e27d60")

    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Object Detection Yield and Class Diversity by Architecture", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{int(h)}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{int(h)}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    b64_url = fig_to_base64(fig)
    saved_path = ""
    if save_to_disk:
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, "graph_object_counts_comparison.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        saved_path = output_path

    plt.close(fig)
    return saved_path, b64_url


def plot_inference_times(
    model_metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    save_to_disk: bool = True
) -> Tuple[str, str]:
    """
    Plot bar chart comparing inference latencies (ms) across all models.

    Returns:
        Tuple of (saved_file_path, base64_data_url).
    """
    models = list(model_metrics.keys())
    latencies = [model_metrics[m]["inference_time_ms"] for m in models]

    min_lat = min(latencies) if latencies else 0
    colors = ["#2e8b57" if lat == min_lat else "#4682b4" for lat in latencies]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, latencies, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Inference Time (milliseconds)", fontsize=12, fontweight="bold")
    ax.set_title("Model Inference Latency Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    b64_url = fig_to_base64(fig)
    saved_path = ""
    if save_to_disk:
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, "graph_inference_times_comparison.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        saved_path = output_path

    plt.close(fig)
    return saved_path, b64_url


def plot_comprehensive_dashboard(
    results_dict: Dict[str, Dict[str, Any]],
    model_metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    save_to_disk: bool = True
) -> Tuple[str, str]:
    """
    Generate an integrated 2x2 analytical visualization dashboard:
    1. Inference Latency (ms)
    2. Object Counts & Class Diversity
    3. Confidence Score Distributions (Boxplot)
    4. Per-Class Category Breakdown

    Returns:
        Tuple of (saved_file_path, base64_data_url).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#f9f9f9")

    models = list(model_metrics.keys())

    # Subplot 1: Inference Latency
    ax1 = axes[0, 0]
    latencies = [model_metrics[m]["inference_time_ms"] for m in models]
    min_lat = min(latencies) if latencies else 0
    colors = ["#1b9e77" if l == min_lat else "#386cb0" for l in latencies]
    bars1 = ax1.bar(models, latencies, color=colors, width=0.5, edgecolor="#333333")
    ax1.set_ylabel("Latency (ms)", fontweight="bold")
    ax1.set_title("Inference Latency by Architecture", fontweight="bold", fontsize=12)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f} ms", (b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

    # Subplot 2: Total Objects & Unique Classes
    ax2 = axes[0, 1]
    x = np.arange(len(models))
    w = 0.35
    t_objs = [model_metrics[m]["total_objects"] for m in models]
    u_cls = [model_metrics[m]["unique_classes_count"] for m in models]
    ax2.bar(x - w / 2, t_objs, w, label="Total Detections", color="#2b5c8f")
    ax2.bar(x + w / 2, u_cls, w, label="Unique Classes", color="#d95f02")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylabel("Count", fontweight="bold")
    ax2.set_title("Detection Count & Diversity", fontweight="bold", fontsize=12)
    ax2.legend(frameon=True)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # Subplot 3: Confidence Score Distributions (Boxplot)
    ax3 = axes[1, 0]
    conf_data = [model_metrics[m]["confidences"] if model_metrics[m]["confidences"] else [0.0] for m in models]
    bp = ax3.boxplot(conf_data, patch_artist=True)
    ax3.set_xticks(range(1, len(models) + 1))
    ax3.set_xticklabels(models, fontweight="bold")
    box_colors = ["#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax3.set_ylabel("Confidence Score", fontweight="bold")
    ax3.set_title("Confidence Score Distribution", fontweight="bold", fontsize=12)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    # Subplot 4: Top Detected Classes across models
    ax4 = axes[1, 1]
    all_classes_union = set()
    for m in models:
        all_classes_union.update(model_metrics[m]["class_distribution"].keys())

    sorted_classes = sorted(list(all_classes_union))[:10]
    if sorted_classes:
        x_cls = np.arange(len(sorted_classes))
        sub_w = 0.8 / len(models)
        for i, m in enumerate(models):
            counts = [model_metrics[m]["class_distribution"].get(c, 0) for c in sorted_classes]
            ax4.bar(x_cls + (i - len(models) / 2 + 0.5) * sub_w, counts, sub_w, label=m)
        ax4.set_xticks(x_cls)
        ax4.set_xticklabels(sorted_classes, rotation=35, ha="right", fontweight="bold")
        ax4.set_ylabel("Frequency", fontweight="bold")
        ax4.set_title("Class Frequency Breakdown Across Models", fontweight="bold", fontsize=12)
        ax4.legend(frameon=True, fontsize=9)
        ax4.grid(axis="y", linestyle=":", alpha=0.6)
    else:
        ax4.text(0.5, 0.5, "No objects detected", ha="center", va="center")

    plt.suptitle("Deep Learning Indoor Object Detection Comparative Dashboard", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    b64_url = fig_to_base64(fig)
    saved_path = ""
    if save_to_disk:
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, "comparative_analysis_dashboard.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        saved_path = output_path

    plt.close(fig)
    return saved_path, b64_url


def generate_observations(
    model_metrics: Dict[str, Any],
    overlap_info: Dict[str, Any],
    save_to_disk: bool = True
) -> str:
    """
    Synthesize detailed, scientific observations analyzing architectural differences,
    inference speeds, detection sensitivity, confidence calibrations, and indoor scene suitability.
    """
    models = list(model_metrics.keys())
    latencies = {m: model_metrics[m]["inference_time_ms"] for m in models}
    counts = {m: model_metrics[m]["total_objects"] for m in models}
    classes_cnt = {m: model_metrics[m]["unique_classes_count"] for m in models}

    fastest_model = min(latencies, key=latencies.get)
    slowest_model = max(latencies, key=latencies.get)
    speedup = latencies[slowest_model] / max(latencies[fastest_model], 0.001)

    highest_detector = max(counts, key=counts.get)
    highest_diversity = max(classes_cnt, key=classes_cnt.get)

    consensus = overlap_info.get("consensus_classes", [])
    consensus_text = ", ".join(consensus) if consensus else "None (differing class assignments or thresholds)"

    lines = [
        "=" * 80,
        "COMPARATIVE ANALYSIS: EXPERT OBSERVATIONS & ARCHITECTURAL EVALUATION",
        "=" * 80,
        "",
        "1. INFERENCE LATENCY & RUNTIME EFFICIENCY:",
        f"   - Fastest Model: {fastest_model} ({latencies[fastest_model]:.1f} ms)",
        f"   - Slowest Model: {slowest_model} ({latencies[slowest_model]:.1f} ms)",
        f"   - Latency Differential: {fastest_model} is {speedup:.1f}x faster than {slowest_model}.",
        "   - Architectural Factor:",
        "     * YOLO11 operates as an anchor-free single-stage CNN with streamlined feature pyamids,",
        "       making it exceptionally well-suited for edge and real-time indoor perception.",
        "     * Faster R-CNN relies on a two-stage Region Proposal Network (RPN) + RoI head,",
        "       introducing multi-step computational overhead.",
        "     * RT-DETR leverages an efficient hybrid encoder and transformer decoder queries to achieve",
        "       end-to-end NMS-free detection while maintaining strong global visual context.",
        "     * Grounding DINO performs cross-modal text-guided self-attention and cross-attention,",
        "       which incurs higher compute per token but unlocks zero-shot open-vocabulary detection.",
        "",
        "2. DETECTION SENSITIVITY & OBJECT DENSITY:",
        f"   - Maximum Objects Detected: {highest_detector} ({counts[highest_detector]} objects)",
        f"   - Maximum Class Diversity:  {highest_diversity} ({classes_cnt[highest_diversity]} distinct classes)",
        f"   - Total Detection Counts across models:",
    ]

    for m in models:
        lines.append(f"     * {m:<16}: {counts[m]} objects across {classes_cnt[m]} unique classes.")

    lines.extend([
        "",
        "3. CONFIDENCE SCORE DISTRIBUTION & CALIBRATION DYNAMICS:",
        "   - Important Technical Note: Confidence scores from these architectures are NOT directly",
        "     equivalent because they originate from fundamentally different probabilistic formulations:",
        "     * Faster R-CNN uses softmax cross-entropy over proposed RoIs.",
        "     * YOLO11 calculates decoupled classification logits multiplied by IoU-aware prediction.",
        "     * RT-DETR utilizes Hungarian matching loss over set prediction queries.",
        "     * Grounding DINO computes dot-product cosine similarities between multimodal text tokens",
        "       and visual bounding query features.",
        "   - Confidence ranges observed:",
    ])

    for m in models:
        met = model_metrics[m]
        lines.append(
            f"     * {m:<16}: Mean = {met['avg_confidence']:.3f} | Min = {met['min_confidence']:.3f} | Max = {met['max_confidence']:.3f} | Std = {met['std_confidence']:.3f}"
        )

    lines.extend([
        "",
        "4. MODEL CONSENSUS & DETECTION OVERLAP:",
        f"   - Universal Consensus (Objects detected by ALL four models):",
        f"     -> {consensus_text}",
        "",
        "   - Objects Unique to a Single Model (Novel Discoveries):",
    ])

    for m in models:
        u_list = overlap_info["unique_to_model"].get(m, [])
        u_str = ", ".join(u_list) if u_list else "None"
        lines.append(f"     * {m:<16}: {u_str}")

    lines.extend([
        "",
        "   - Objects Missed by Individual Models (Detected by others):",
    ])

    for m in models:
        m_list = overlap_info["missed_by_model"].get(m, [])
        m_str = ", ".join(m_list) if m_list else "None (captured all classes identified by others)"
        lines.append(f"     * {m:<16}: {m_str}")

    lines.extend([
        "",
        "5. PRACTICAL RECOMMENDATIONS FOR INDOOR COMPUTER VISION:",
        "   - Real-Time Robotics & Edge Deployment:",
        "     * YOLO11 and RT-DETR offer the optimal latency-accuracy Pareto frontier for robots and smart cameras.",
        "   - High-Precision Spatial Layout & Forensic Scene Understanding:",
        "     * Grounding DINO allows detecting arbitrary customized indoor objects (e.g. specific furniture,",
        "       tools, or uncommon appliances) without retraining or fine-tuning weights.",
        "     * Faster R-CNN provides reliable bounding box localization on standard COCO indoor categories.",
        "=" * 80
    ])

    report = "\n".join(lines)

    if save_to_disk:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_path = os.path.join(OUTPUT_DIR, "comparison_observations.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def run_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    original_image_path: Optional[str] = None,
    save_to_disk: bool = True
) -> Dict[str, Any]:
    """
    Main entry point for running the complete comparative analysis.

    Returns dictionary containing metrics, overlap analysis, table records,
    base64 visualization charts, and observation reports.
    """
    # 1. Compute metrics and overlap
    model_metrics, overlap_info = compute_metrics(results_dict)

    # 2. Build Pandas comparison table
    df_table = generate_comparison_table(model_metrics, save_to_disk=save_to_disk)

    # 3. Generate Visualizations (both files and base64 URLs)
    grid_path, grid_b64 = plot_2x2_grid(results_dict, save_to_disk=save_to_disk)
    counts_path, counts_b64 = plot_object_counts(model_metrics, save_to_disk=save_to_disk)
    latency_path, latency_b64 = plot_inference_times(model_metrics, save_to_disk=save_to_disk)
    dashboard_path, dashboard_b64 = plot_comprehensive_dashboard(results_dict, model_metrics, save_to_disk=save_to_disk)

    # 4. Generate observations
    observations_text = generate_observations(model_metrics, overlap_info, save_to_disk=save_to_disk)

    return {
        "metrics": model_metrics,
        "overlap_info": overlap_info,
        "table_dataframe": df_table,
        "table_records": df_table.to_dict(orient="records"),
        "grid_image_path": grid_path,
        "grid_image_base64": grid_b64,
        "counts_chart_path": counts_path,
        "counts_chart_base64": counts_b64,
        "latency_chart_path": latency_path,
        "latency_chart_base64": latency_b64,
        "dashboard_path": dashboard_path,
        "dashboard_base64": dashboard_b64,
        "observations": observations_text
    }
