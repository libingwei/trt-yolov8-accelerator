#!/usr/bin/env python3
"""
Dataset Validation Script
Validates downloaded datasets and provides usage examples
"""

import os
import sys
from pathlib import Path
import argparse

def validate_dataset(data_dir):
    """Validate downloaded datasets"""
    data_dir = Path(data_dir)
    
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    
    datasets = {
        "COCO Val2017": {
            "path": data_dir / "coco" / "val2017",
            "expected_files": 5000,
            "file_patterns": ["*.jpg", "*.jpeg"]
        },
        "COCO Annotations": {
            "path": data_dir / "coco" / "annotations",
            "expected_files": 5,
            "file_patterns": ["*.json"]
        },
        "YOLOv8 Test Images": {
            "path": data_dir / "yolov8_test", 
            "expected_files": 2,
            "file_patterns": ["*.jpg", "*.png"]
        },
        "ImageNet Sample": {
            "path": data_dir / "imagenet_sample",
            "expected_files": 5,  # Flexible
            "file_patterns": ["*.jpg", "*.jpeg", "*.png"]
        },
        "COCO Calibration": {
            "path": data_dir / "calibration" / "coco_subset",
            "expected_files": 100,
            "file_patterns": ["*.jpg", "*.jpeg"]
        },
        "ImageNet Calibration": {
            "path": data_dir / "calibration" / "imagenet_subset", 
            "expected_files": 100,
            "file_patterns": ["*.jpg", "*.jpeg", "*.png"]
        }
    }
    
    valid_datasets = []
    
    for name, config in datasets.items():
        path = config["path"]
        expected = config["expected_files"]
        patterns = config["file_patterns"]
        
        if not path.exists():
            print(f"❌ {name}: Directory not found ({path})")
            continue
            
        # Count files matching patterns
        file_count = 0
        for pattern in patterns:
            file_count += len(list(path.glob(pattern)))
            
        if file_count == 0:
            print(f"❌ {name}: No files found ({path})")
        elif file_count < expected * 0.8:  # Allow some tolerance
            print(f"⚠️  {name}: {file_count} files (expected ~{expected}) at {path}")
            valid_datasets.append((name, path, file_count))
        else:
            print(f"✅ {name}: {file_count} files at {path}")
            valid_datasets.append((name, path, file_count))
    
    return valid_datasets

def show_usage_examples(data_dir, valid_datasets):
    """Show usage examples based on available datasets"""
    data_dir = Path(data_dir)
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    # Find example images
    example_images = []
    test_dir = data_dir / "yolov8_test"
    if test_dir.exists():
        example_images.extend(list(test_dir.glob("*.jpg"))[:2])
    
    coco_dir = data_dir / "coco" / "val2017"
    if coco_dir.exists():
        example_images.extend(list(coco_dir.glob("*.jpg"))[:1])
    
    # Find calibration directories
    calib_dirs = []
    calib_base = data_dir / "calibration"
    if calib_base.exists():
        calib_dirs = [d for d in calib_base.iterdir() if d.is_dir()]
    
    print("\n1. Build TensorRT Engine:")
    if calib_dirs:
        print(f"   # FP16 build")
        print(f"   ./build/bin/onnx_to_trt_yolo model.onnx engine_fp16.trt --fp16")
        print(f"   ")
        print(f"   # INT8 build with calibration")
        print(f"   ./build/bin/onnx_to_trt_yolo model.onnx engine_int8.trt --int8 \\")
        print(f"       --calib-dir {calib_dirs[0]}")
        print(f"   ")
        print(f"   # Plugin-enabled build")
        print(f"   ./build/bin/onnx_to_trt_yolo model.onnx engine_plugin.trt --fp16 \\")
        print(f"       --decode-plugin --nms-plugin")
    else:
        print(f"   ./build/bin/onnx_to_trt_yolo model.onnx engine.trt --fp16")
    
    print(f"\n2. Run Inference:")
    if example_images:
        for img in example_images[:3]:
            print(f"   ./build/bin/yolo_trt_infer engine.trt --image {img}")
        
        print(f"   ")
        print(f"   # With plugin decoding")
        print(f"   ./build/bin/yolo_trt_infer engine_plugin.trt --image {example_images[0]} \\")
        print(f"       --decode plugin --has-nms")
    
    print(f"\n3. Batch Processing:")
    if coco_dir.exists():
        print(f"   # Process first 10 COCO images")
        print(f"   for img in {coco_dir}/*.jpg | head -10; do")
        print(f"       ./build/bin/yolo_trt_infer engine.trt --image \"$img\"")
        print(f"   done")
    
    print(f"\n4. Download More Data:")
    print(f"   # Download additional calibration images")
    print(f"   python scripts/download_yolov8_datasets.py --calib-images 200")
    print(f"   ")
    print(f"   # Download to custom directory") 
    print(f"   python scripts/download_yolov8_datasets.py --data-dir /custom/path")

def main():
    parser = argparse.ArgumentParser(description="Validate YOLOv8 datasets")
    parser.add_argument("--data-dir", type=str, default="./datasets",
                        help="Dataset directory to validate")
    parser.add_argument("--download-missing", action="store_true",
                        help="Download missing datasets")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    
    if not data_dir.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        print(f"Run: python scripts/download_yolov8_datasets.py --data-dir {data_dir}")
        return 1
    
    # Validate existing datasets
    valid_datasets = validate_dataset(data_dir)
    
    # Show usage examples
    show_usage_examples(data_dir, valid_datasets)
    
    # Offer to download missing datasets
    if args.download_missing and len(valid_datasets) < 4:
        print(f"\n" + "=" * 60)
        print("DOWNLOADING MISSING DATASETS")
        print("=" * 60)
        
        import subprocess
        cmd = [sys.executable, "scripts/download_yolov8_datasets.py", 
               "--data-dir", str(data_dir), "--dataset", "all"]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n✅ Download completed. Re-validating...")
            validate_dataset(data_dir)
        else:
            print("\n❌ Download failed")
    
    print(f"\n" + "=" * 60)
    print(f"Dataset validation completed for: {data_dir}")
    print(f"Valid datasets: {len(valid_datasets)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
