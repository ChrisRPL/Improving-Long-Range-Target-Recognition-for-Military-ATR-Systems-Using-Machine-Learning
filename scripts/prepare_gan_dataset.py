import os
import cv2
from utils.video_processing import extract_frames
from utils.annotations import load_yolo_annotations

def prepare_gan_dataset(video_path, annotation_path, output_dir, crop_size=64):
    frames = extract_frames(video_path)
    annotations = load_yolo_annotations(annotation_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_number, (class_id, x_center, y_center, width, height) in enumerate(annotations):
        frame = frames[frame_number]
        h, w = frame.shape[:2]
        x1 = int((x_center - width / 2) * w)
        y1 = int((x_center + width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        y2 = int((y_center + height / 2) * h)
        
        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (crop_size, crop_size))
        
        output_path = os.path.join(output_dir, f'{frame_number}_{class_id}.png')
        cv2.imwrite(output_path, crop_resized)
