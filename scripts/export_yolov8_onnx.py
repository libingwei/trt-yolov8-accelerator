#!/usr/bin/env python3
"""
Export YOLOv8 to ONNX with dynamic shapes and verify with onnxruntime.
"""
import argparse
import onnx
import onnxruntime as ort
from pathlib import Path
import sys
try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("ultralytics not available. pip install -r requirements.txt")


def export(weights: str, outdir: str, opset: int = 12, dynamic: bool = True, imgsz: int = 640, device: str = "auto"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / (Path(weights).stem + ".onnx")

    model = YOLO(weights)
    # Resolve device preference: auto -> prefer CUDA then CPU
    if device == "auto":
        use_cuda = torch is not None and torch.cuda.is_available()
        dev = "cuda:0" if use_cuda else "cpu"
    else:
        dev = device

    # Try export on preferred device; fall back to CPU if it fails
    try:
        model.export(format="onnx", opset=opset, dynamic=dynamic, imgsz=imgsz, optimize=True, device=dev)
    except Exception as e:
        if dev.startswith("cuda"):
            print(f"[warn] CUDA export failed on {dev}, falling back to CPU. Error: {e}", file=sys.stderr)
            model.export(format="onnx", opset=opset, dynamic=dynamic, imgsz=imgsz, optimize=True, device="cpu")
        else:
            raise
    # ultralytics exports next to weights; move if needed
    default_out = Path(weights).with_suffix(".onnx")
    if default_out.exists() and default_out != onnx_path:
        default_out.replace(onnx_path)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]) 
    inp = sess.get_inputs()[0]
    print("Exported:", onnx_path)
    print("Input:", inp.name, inp.shape, inp.type)
    return str(onnx_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-dynamic", action="store_true")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda:0 (default: auto, prefer GPU then fallback CPU)")
    args = ap.parse_args()
    export(args.weights, args.outdir, args.opset, not args.no_dynamic, args.imgsz, args.device)
