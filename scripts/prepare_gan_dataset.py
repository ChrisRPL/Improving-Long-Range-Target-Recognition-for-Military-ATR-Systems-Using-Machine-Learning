import os
import cv2
from utils.video_processing import extract_frames
from utils.annotations import load_coco_annotations

def prepare_gan_dataset(video_path, annotation_path, low_res_output_dir, high_res_output_dir, crop_size=64):
    frames, filenames = extract_frames(video_path)
    annotations, id_to_filename = load_coco_annotations(annotation_path)
   
    for img_id, anns in annotations.items():
        filename = id_to_filename[img_id]
        if filename not in filenames:
            continue
           
        frame_idx = filenames.index(filename)
        frame = frames[frame_idx]
       
        for category_id, bbox in anns:
            x, y, w, h = [int(c) for c in bbox]
            crop = frame[y:y+h, x:x+w]
            crop_resized = cv2.resize(crop, (crop_size, crop_size))
           
            low_res_output_path = os.path.join(low_res_output_dir, f'{filename}_{category_id}.png')
            high_res_output_path = os.path.join(high_res_output_dir, f'{filename}_{category_id}.png')
            cv2.imwrite(low_res_output_path, crop_resized)
            cv2.imwrite(high_res_output_path, crop)
