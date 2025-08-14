#!/bin/bash

# Video Tracking Demo Script
# This script demonstrates how to use the YOLOv8 video tracking system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}YOLOv8 Video Tracking Demo${NC}"
echo "================================"

# Check if engine file exists
ENGINE_PATH="$PROJECT_DIR/models/yolov8n_fp16.trt"
if [ ! -f "$ENGINE_PATH" ]; then
    echo -e "${RED}Error: TensorRT engine not found at $ENGINE_PATH${NC}"
    echo "Please first convert ONNX to TensorRT engine using:"
    echo "  ./build/bin/onnx_to_trt_yolo models/yolov8n.onnx models/yolov8n_fp16.trt --fp16"
    exit 1
fi

# Check if executable exists
TRACKER_EXE="$PROJECT_DIR/build/bin/video_tracker"
if [ ! -f "$TRACKER_EXE" ]; then
    echo -e "${RED}Error: Video tracker executable not found at $TRACKER_EXE${NC}"
    echo "Please build the project first:"
    echo "  cd build && make -j\$(nproc)"
    exit 1
fi

# Create test videos directory if it doesn't exist
TEST_VIDEOS_DIR="$PROJECT_DIR/video_tracking/test_videos"
mkdir -p "$TEST_VIDEOS_DIR"

echo -e "${YELLOW}Available options:${NC}"
echo "1. Process sample video file"
echo "2. Process camera stream"
echo "3. Create a simple test video"
echo ""

read -p "Choose an option (1-3): " choice

case $choice in
    1)
        echo -e "${GREEN}Processing video file...${NC}"
        
        # Look for sample video
        SAMPLE_VIDEO=""
        for ext in mp4 avi mov mkv; do
            for file in "$TEST_VIDEOS_DIR"/*.$ext; do
                if [ -f "$file" ]; then
                    SAMPLE_VIDEO="$file"
                    break 2
                fi
            done
        done
        
        if [ -z "$SAMPLE_VIDEO" ]; then
            echo -e "${YELLOW}No sample video found in $TEST_VIDEOS_DIR${NC}"
            read -p "Enter path to video file: " SAMPLE_VIDEO
            
            if [ ! -f "$SAMPLE_VIDEO" ]; then
                echo -e "${RED}Video file not found: $SAMPLE_VIDEO${NC}"
                exit 1
            fi
        fi
        
        OUTPUT_VIDEO="$TEST_VIDEOS_DIR/tracked_output.mp4"
        
        echo "Input video: $SAMPLE_VIDEO"
        echo "Output video: $OUTPUT_VIDEO"
        echo ""
        
        "$TRACKER_EXE" "$ENGINE_PATH" \
            --video "$SAMPLE_VIDEO" \
            --output "$OUTPUT_VIDEO" \
            --conf 0.25 \
            --iou 0.5
        
        echo -e "${GREEN}Processing completed! Output saved to: $OUTPUT_VIDEO${NC}"
        ;;
        
    2)
        echo -e "${GREEN}Starting camera stream...${NC}"
        read -p "Enter camera ID (default: 0): " camera_id
        camera_id=${camera_id:-0}
        
        read -p "Save output? (y/N): " save_output
        
        if [[ $save_output =~ ^[Yy]$ ]]; then
            OUTPUT_VIDEO="$TEST_VIDEOS_DIR/camera_tracked_$(date +%Y%m%d_%H%M%S).mp4"
            echo "Output will be saved to: $OUTPUT_VIDEO"
            
            "$TRACKER_EXE" "$ENGINE_PATH" \
                --camera "$camera_id" \
                --output "$OUTPUT_VIDEO" \
                --conf 0.25 \
                --iou 0.5
        else
            "$TRACKER_EXE" "$ENGINE_PATH" \
                --camera "$camera_id" \
                --conf 0.25 \
                --iou 0.5
        fi
        ;;
        
    3)
        echo -e "${GREEN}Creating simple test video...${NC}"
        
        # Check if we have ffmpeg
        if ! command -v ffmpeg &> /dev/null; then
            echo -e "${RED}ffmpeg is required to create test video${NC}"
            echo "Please install ffmpeg and try again"
            exit 1
        fi
        
        TEST_VIDEO="$TEST_VIDEOS_DIR/test_video.mp4"
        
        echo "Creating a simple test video with moving objects..."
        
        # Create a simple test video using ffmpeg
        ffmpeg -f lavfi -i "testsrc2=duration=10:size=640x480:rate=30" \
               -f lavfi -i "color=red:size=50x50:duration=10:rate=30" \
               -filter_complex "[1]scale=50:50[box];[0][box]overlay=x='if(gte(t,1),100+50*t,NAN)':y=200" \
               -c:v libx264 -t 10 -y "$TEST_VIDEO"
        
        echo -e "${GREEN}Test video created: $TEST_VIDEO${NC}"
        echo "Now processing the test video..."
        
        OUTPUT_VIDEO="$TEST_VIDEOS_DIR/test_tracked_output.mp4"
        
        "$TRACKER_EXE" "$ENGINE_PATH" \
            --video "$TEST_VIDEO" \
            --output "$OUTPUT_VIDEO" \
            --conf 0.25 \
            --iou 0.5
        
        echo -e "${GREEN}Test completed! Output saved to: $OUTPUT_VIDEO${NC}"
        ;;
        
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Demo completed successfully!${NC}"
