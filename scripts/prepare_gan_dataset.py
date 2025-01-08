import os
import cv2
from utils.video_processing import extract_frames
from utils.annotations import load_coco_annotations
import sys
from pathlib import Path
import argparse
import json
sys.path.append(str(Path(__file__).parent.parent))

def crop_square(frame, x, y, w, h):
    # Find the center of the detected object
    center_x = x + w//2
    center_y = y + h//2
    
    # Use the larger dimension for the square
    size = max(w, h)
    
    # Calculate square boundaries keeping the object centered
    x1 = max(0, center_x - size//2)
    y1 = max(0, center_y - size//2)
    x2 = min(frame.shape[1], x1 + size)
    y2 = min(frame.shape[0], y1 + size)
    
    # Adjust if square goes beyond frame boundaries
    if x1 < 0:
        x2 = min(frame.shape[1], size)
        x1 = 0
    if y1 < 0:
        y2 = min(frame.shape[0], size)
        y1 = 0
    
    # Ensure we get a square crop even at image boundaries
    actual_size = min(x2 - x1, y2 - y1)
    x2 = x1 + actual_size
    y2 = y1 + actual_size
    
    return frame[int(y1):int(y2), int(x1):int(x2)]

def prepare_gan_dataset(video_path, annotation_path, low_res_output_dir, high_res_output_dir, matching_dict_path, crop_size=64):
    annotations, id_to_filename = load_coco_annotations(annotation_path)
    with open(matching_dict_path) as json_file:
        data = json.load(json_file)
    
    filenames = list(data.values())
    print("preparing dataset")
    
   
    for img_id, anns in annotations.items():
        filename = id_to_filename[img_id].replace("data/", "")
        print(filename)
        if filename not in filenames:
            continue
        
        frame_idx = filenames.index(filename)
        frame = cv2.imread(os.path.join(video_path, filename))
        
        if frame is None:
            print(f"Warning: Could not read frame {filename}")
            continue
        
        for category_id, bbox in anns:
            x, y, w, h = [int(c) for c in bbox]
            
            # Get square crop
            crop = crop_square(frame, x, y, w, h)
            
            # Skip if crop is empty
            if crop.size == 0:
                print(f"Warning: Empty crop for {filename}_{category_id}")
                continue
            
            # Create low-res version
            crop_resized = cv2.resize(crop, (crop_size, crop_size))
            
            # Save both versions
            low_res_output_path = os.path.join(low_res_output_dir, f'{filename}_{category_id}.png')
            high_res_output_path = os.path.join(high_res_output_dir, f'{filename}_{category_id}.png')
            
            cv2.imwrite(low_res_output_path, crop_resized)
            cv2.imwrite(high_res_output_path, crop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Fusion Dataset")
    parser.add_argument('--video_frames_path', type=str, required=True, help='Path to the video frames')
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to coco json annotations')
    parser.add_argument('--low_res_output_dir', type=str, required=True, help='Path to low res output images')
    parser.add_argument('--high_res_output_dir', type=str, required=True, help='Path to high res output images')
    parser.add_argument('--matching_dict_path', type=str, required=True, help='Path to json of rgb-thermal pairs')

    args = parser.parse_args()

    prepare_gan_dataset(args.video_frames_path, args.annotation_path, args.low_res_output_dir, args.high_res_output_dir, args.matching_dict_path)
