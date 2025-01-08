import os
import cv2
from utils.video_processing import extract_frames
from utils.annotations import load_coco_annotations
import sys
from pathlib import Path
import argparse
sys.path.append(str(Path(__file__).parent.parent))

def prepare_gan_dataset(video_path, annotation_path, low_res_output_dir, high_res_output_dir, matching_dict_path, crop_size=32):
    annotations, id_to_filename = load_coco_annotations(annotation_path)
   
    for img_id, anns in annotations.items():
        filename = id_to_filename[img_id]
        if filename not in filenames:
            continue
           
        frame_idx = filenames.index(filename)
        frame = cv2.imread(os.path.join(filename, video_path))
       
        for category_id, bbox in anns:
            x, y, w, h = [int(c) for c in bbox]
            crop = frame[y:y+h, x:x+w]
            crop_resized = cv2.resize(crop, (crop_size, crop_size))
           
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
