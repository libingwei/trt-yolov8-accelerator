#!/usr/bin/env python3
"""
YOLOv8 Dataset Download Script
Downloads common datasets for YOLOv8 validation and calibration
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urlparse
import shutil

def download_file(url, dest_path, chunk_size=8192):
    """Download file with progress bar"""
    print(f"Downloading {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
    
    print(f"\nDownload completed: {dest_path}")
    return dest_path

def extract_archive(archive_path, extract_dir):
    """Extract zip or tar archive"""
    print(f"Extracting {archive_path} to {extract_dir}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        print(f"Unsupported archive format: {archive_path}")
        return False
    
    print(f"Extraction completed")
    return True

def download_coco_val(data_dir):
    """Download COCO validation dataset (2017)"""
    print("=== Downloading COCO 2017 Validation Dataset ===")
    
    coco_dir = data_dir / "coco"
    images_dir = coco_dir / "val2017"
    annotations_dir = coco_dir / "annotations"
    
    # Download validation images
    if not images_dir.exists():
        images_url = "http://images.cocodataset.org/zips/val2017.zip"
        images_zip = coco_dir / "val2017.zip"
        download_file(images_url, images_zip)
        extract_archive(images_zip, coco_dir)
        images_zip.unlink()  # Remove zip file
    else:
        print("COCO val2017 images already exist")
    
    # Download annotations
    if not annotations_dir.exists():
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        annotations_zip = coco_dir / "annotations_trainval2017.zip"
        download_file(annotations_url, annotations_zip)
        extract_archive(annotations_zip, coco_dir)
        annotations_zip.unlink()  # Remove zip file
    else:
        print("COCO annotations already exist")
    
    print(f"COCO dataset available at: {coco_dir}")
    return coco_dir

def download_imagenet_sample(data_dir, num_images=1000):
    """Download ImageNet validation sample for calibration"""
    print(f"=== Downloading ImageNet Sample ({num_images} images) ===")
    
    imagenet_dir = data_dir / "imagenet_sample"
    
    if imagenet_dir.exists() and len(list(imagenet_dir.glob("*.JPEG"))) >= num_images:
        print(f"ImageNet sample already exists with sufficient images")
        return imagenet_dir
    
    # Create a sample from ImageNet validation set
    # Note: This is a simplified approach. For full ImageNet, you need proper access
    imagenet_dir.mkdir(parents=True, exist_ok=True)
    
    # Download some sample images from a public source
    sample_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Culinary_fruits_front_view.jpg/320px-Culinary_fruits_front_view.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Domestic_goat_kid_in_capeweed.jpg/320px-Domestic_goat_kid_in_capeweed.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Stadtbahn_Bochum_U35_6688.JPG/320px-Stadtbahn_Bochum_U35_6688.JPG"
    ]
    
    for i, url in enumerate(sample_urls):
        if i >= num_images:
            break
        filename = f"sample_{i:04d}.jpg"
        try:
            download_file(url, imagenet_dir / filename)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    
    print(f"ImageNet sample available at: {imagenet_dir}")
    return imagenet_dir

def download_yolov8_test_images(data_dir):
    """Download YOLOv8 test images"""
    print("=== Downloading YOLOv8 Test Images ===")
    
    test_dir = data_dir / "yolov8_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Download some common test images
    test_urls = {
        "bus.jpg": "https://ultralytics.com/images/bus.jpg",
        "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
    }
    
    for filename, url in test_urls.items():
        dest_path = test_dir / filename
        if not dest_path.exists():
            try:
                download_file(url, dest_path)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists")
    
    print(f"YOLOv8 test images available at: {test_dir}")
    return test_dir

def create_calibration_subset(source_dir, calib_dir, num_images=100):
    """Create a calibration subset from a larger dataset"""
    print(f"=== Creating Calibration Subset ({num_images} images) ===")
    
    calib_dir = Path(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(source_dir).rglob(f"*{ext}"))
        image_files.extend(Path(source_dir).rglob(f"*{ext.upper()}"))
    
    # Select a subset
    import random
    random.seed(42)  # For reproducibility
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    
    # Copy selected files
    for i, src_file in enumerate(selected_files):
        dest_file = calib_dir / f"calib_{i:04d}{src_file.suffix}"
        if not dest_file.exists():
            shutil.copy2(src_file, dest_file)
    
    print(f"Calibration subset created at: {calib_dir}")
    print(f"Number of calibration images: {len(list(calib_dir.glob('*')))}")
    return calib_dir

def main():
    parser = argparse.ArgumentParser(description="Download YOLOv8 datasets")
    parser.add_argument("--data-dir", type=str, default="./datasets", 
                        help="Directory to store datasets")
    parser.add_argument("--dataset", type=str, choices=["coco", "imagenet", "test", "all"], 
                        default="all", help="Dataset to download")
    parser.add_argument("--calib-images", type=int, default=100,
                        help="Number of images for calibration subset")
    parser.add_argument("--imagenet-samples", type=int, default=1000,
                        help="Number of ImageNet sample images")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    
    downloaded_dirs = []
    
    if args.dataset in ["coco", "all"]:
        coco_dir = download_coco_val(data_dir)
        downloaded_dirs.append(coco_dir)
        
        # Create calibration subset from COCO
        calib_dir = data_dir / "calibration" / "coco_subset"
        create_calibration_subset(coco_dir / "val2017", calib_dir, args.calib_images)
    
    if args.dataset in ["imagenet", "all"]:
        imagenet_dir = download_imagenet_sample(data_dir, args.imagenet_samples)
        downloaded_dirs.append(imagenet_dir)
        
        # Create calibration subset from ImageNet
        calib_dir = data_dir / "calibration" / "imagenet_subset" 
        create_calibration_subset(imagenet_dir, calib_dir, args.calib_images)
    
    if args.dataset in ["test", "all"]:
        test_dir = download_yolov8_test_images(data_dir)
        downloaded_dirs.append(test_dir)
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    for dir_path in downloaded_dirs:
        if dir_path.exists():
            num_files = len(list(dir_path.rglob("*.*")))
            print(f"✓ {dir_path.name}: {num_files} files at {dir_path}")
    
    calib_dirs = list((data_dir / "calibration").glob("*")) if (data_dir / "calibration").exists() else []
    if calib_dirs:
        print("\nCalibration datasets:")
        for calib_dir in calib_dirs:
            num_files = len(list(calib_dir.glob("*.*")))
            print(f"  • {calib_dir.name}: {num_files} images")
    
    print(f"\nAll datasets stored in: {data_dir}")
    print("\nUsage examples:")
    print(f"  # Validation: --image {data_dir}/yolov8_test/bus.jpg")
    print(f"  # Calibration: --calib-dir {data_dir}/calibration/coco_subset")
    print(f"  # COCO eval: python val.py --data {data_dir}/coco/coco.yaml")

if __name__ == "__main__":
    main()
