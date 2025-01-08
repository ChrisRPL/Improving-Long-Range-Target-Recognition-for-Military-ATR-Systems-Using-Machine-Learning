import os
import cv2
import json
import argparse
from pathlib import Path
import numpy as np
from typing import Tuple, Dict

def load_coco_annotations(annotation_path: str) -> Tuple[Dict, Dict]:
    """Load COCO annotations and return image annotations and id mapping."""
    with open(annotation_path) as f:
        coco_data = json.load(f)
    
    # Create image id to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Group annotations by image id
    annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append((ann['category_id'], ann['bbox']))
    
    return annotations, id_to_filename

def crop_square(frame, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop a square region around object, maintaining aspect ratio."""
    center_x = x + w//2
    center_y = y + h//2
    size = max(w, h)
    
    # Calculate square boundaries
    x1 = max(0, center_x - size//2)
    y1 = max(0, center_y - size//2)
    x2 = min(frame.shape[1], x1 + size)
    y2 = min(frame.shape[0], y1 + size)
    
    # Adjust for frame boundaries
    if x1 < 0:
        x2 = min(frame.shape[1], size)
        x1 = 0
    if y1 < 0:
        y2 = min(frame.shape[0], size)
        y1 = 0
    
    # Ensure square crop
    actual_size = min(x2 - x1, y2 - y1)
    x2 = x1 + actual_size
    y2 = y1 + actual_size
    
    return frame[int(y1):int(y2), int(x1):int(x2)]

def prepare_gan_dataset(
    frames_dir: str,
    annotation_path: str,
    output_dir: str,
    matching_dict_path: str,
    target_size: int = 32,
    min_size: int = 96
) -> None:
    """
    Prepare dataset for GAN training by creating high-res and low-res pairs.
    """
    # Create output directories
    output_dir = Path(output_dir)
    high_res_dir = output_dir / 'high_res'
    low_res_dir = output_dir / 'low_res'
    high_res_dir.mkdir(parents=True, exist_ok=True)
    low_res_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations and matching dictionary
    annotations, id_to_filename = load_coco_annotations(annotation_path)
    with open(matching_dict_path) as f:
        matching_dict = json.load(f)
    
    valid_filenames = set(matching_dict.keys())
    processed_count = 0
    
    print("Processing frames...")
    for img_id, anns in annotations.items():
        filename = id_to_filename[img_id].replace("data/", "")
        if filename not in valid_filenames:
            continue
            
        frame_path = Path(frames_dir) / filename
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: Could not read frame {filename}")
            continue
            
        for category_id, bbox in anns:
            x, y, w, h = map(int, bbox)
            
            # Skip small objects
            if w * h < min_size * min_size:
                continue
                
            # Get square crop
            crop = crop_square(frame, x, y, w, h)
            if crop.size == 0:
                continue
                
            # Create high-res and low-res versions
            low_res = cv2.resize(crop, (target_size, target_size))
            
            # Fix: Convert filename to Path before accessing stem and suffix
            path_obj = Path(filename)
            base_name = f'{path_obj.stem}_{category_id}{path_obj.suffix}'
            
            cv2.imwrite(str(low_res_dir / base_name), low_res)
            cv2.imwrite(str(high_res_dir / base_name), crop)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} objects")
    
    print(f"Dataset creation complete. Total objects processed: {processed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GAN Training Dataset")
    parser.add_argument('--frames_dir', type=str, required=True,
                      help='Directory containing input frames')
    parser.add_argument('--annotation_path', type=str, required=True,
                      help='Path to COCO format annotations')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Base directory for output (will create high_res and low_res subdirs)')
    parser.add_argument('--matching_dict_path', type=str, required=True,
                      help='Path to frame matching dictionary')
    parser.add_argument('--target_size', type=int, default=32,
                      help='Size of low-res output (default: 32)')
    parser.add_argument('--min_size', type=int, default=96,
                      help='Minimum size of source objects to use (default: 96)')
    
    args = parser.parse_args()
    prepare_gan_dataset(
        args.frames_dir,
        args.annotation_path,
        args.output_dir,
        args.matching_dict_path,
        args.target_size,
        args.min_size
    )
