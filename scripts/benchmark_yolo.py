"""
ORBITA YOLO Model Benchmark Script (Section 40)
==============================================
Profiles standalone YOLO inference throughput, latency distribution (Avg, P50, P95, P99),
cold start, steady-state, resolution sweep, and precision modes (FP32 vs FP16).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch


def benchmark_yolo(
    model_path: str = "models/orbita_yolo_detector_v4.pt",
    resolutions: list[int] | None = None,
    device: str | None = None,
    half: bool = False,
    warmup: int = 20,
    iterations: int = 150,
):
    if resolutions is None:
        resolutions = [640, 512, 480, 384]

    from ultralytics import YOLO

    resolved_path = Path(model_path)
    if not resolved_path.exists():
        fallback_candidates = [
            Path("models/orbita_yolo_detector_v4.pt"),
            Path("models/orbita_yolo_detector_v3.pt"),
            Path("models/yolov8n.pt"),
        ]
        for c in fallback_candidates:
            if c.exists():
                resolved_path = c
                break

    if not resolved_path.exists():
        print(f"Error: Model not found at {model_path} or fallbacks.")
        return

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 68)
    print("ORBITA YOLO BENCHMARK (Section 40)")
    print("=" * 68)
    print(f"Model:       {resolved_path.name} ({resolved_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Device:      {device.upper()} {'(' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else ''}")
    print(f"Precision:   {'FP16 (Half)' if half and device == 'cuda' else 'FP32'}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"Warmup:      {warmup} iterations")
    print(f"Iterations:  {iterations} iterations per resolution")
    print("=" * 68)

    # 1. Cold-start load measurement
    t_load_start = time.perf_counter()
    model = YOLO(str(resolved_path))
    model.to(device)
    load_time_ms = (time.perf_counter() - t_load_start) * 1000.0

    # 2. Cold-start inference measurement
    dummy_init = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    t_cold_start = time.perf_counter()
    pred_kwargs = {"device": device, "verbose": False, "stream": False}
    if half and device == "cuda":
        pred_kwargs["half"] = True

    with torch.inference_mode():
        _ = model.predict(dummy_init, imgsz=480, **pred_kwargs)
    cold_infer_ms = (time.perf_counter() - t_cold_start) * 1000.0

    print(f"Model Load Time:        {load_time_ms:6.1f} ms")
    print(f"Cold-Start Inference:   {cold_infer_ms:6.1f} ms")
    print("-" * 68)

    # Resolution sweep
    results_summary = []

    for res in resolutions:
        # Generate dummy input frame
        img = np.random.randint(0, 255, (res, res, 3), dtype=np.uint8)

        # Warmup
        for _ in range(warmup):
            with torch.inference_mode():
                _ = model.predict(img, imgsz=res, **pred_kwargs)

        if device == "cuda":
            torch.cuda.synchronize()

        # Timed benchmark iterations
        latencies = []
        t_bench_start = time.perf_counter()

        for _ in range(iterations):
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model.predict(img, imgsz=res, **pred_kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

        total_wall_sec = time.perf_counter() - t_bench_start
        fps = iterations / total_wall_sec

        avg_lat = float(np.mean(latencies))
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        gpu_mem_mb = 0.0
        if device == "cuda":
            gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        results_summary.append({
            "resolution": f"{res}x{res}",
            "avg": avg_lat,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "fps": fps,
            "vram_mb": gpu_mem_mb,
        })

        print(
            f"Res: {res:3d}x{res:3d} | "
            f"Avg: {avg_lat:5.1f} ms | "
            f"P50: {p50:5.1f} ms | "
            f"P95: {p95:5.1f} ms | "
            f"P99: {p99:5.1f} ms | "
            f"FPS: {fps:5.1f}"
            + (f" | VRAM: {gpu_mem_mb:5.1f} MB" if device == "cuda" else "")
        )

    print("=" * 68)
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORBITA YOLO Benchmark")
    parser.add_argument("--model", type=str, default="models/orbita_yolo_detector_v4.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--half", action="store_true", help="Use FP16 precision if on CUDA")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    benchmark_yolo(
        model_path=args.model,
        device=args.device,
        half=args.half,
        warmup=args.warmup,
        iterations=args.iterations,
    )
