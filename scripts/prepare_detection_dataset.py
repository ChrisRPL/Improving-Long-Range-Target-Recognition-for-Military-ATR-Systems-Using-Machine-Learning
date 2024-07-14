# scripts/prepare_detection_dataset.py
import os
import cv2
import argparse
from utils.video_processing import extract_frames, calculate_optical_flow
from utils.annotations import load_yolo_annotations, save_annotations

def prepare_detection_dataset(video_path, annotation_path, output_image_dir, output_flow_dir, output_annotation_dir):
    frames = extract_frames(video_path)
    annotations = load_yolo_annotations(annotation_path)
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_flow_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)

    # Save frames and annotations
    for i, frame in enumerate(frames):
        image_path = os.path.join(output_image_dir, f'frame_{i}.png')
        cv2.imwrite(image_path, frame)
        
        annotation = annotations[i]
        annotation_path = os.path.join(output_annotation_dir, f'frame_{i}.txt')
        save_annotations([annotation], annotation_path)

    # Calculate and save optical flows
    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        next_frame = frames[i]
        flow = calculate_optical_flow(prev_frame, next_frame)
        
        flow_path = os.path.join(output_flow_dir, f'flow_{i}.npy')
        np.save(flow_path, flow)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Detection Dataset")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to the annotation file')
    parser.add_argument('--output_image_dir', type=str, required=True, help='Directory to save extracted frames')
    parser.add_argument('--output_flow_dir', type=str, required=True, help='Directory to save optical flow images')
    parser.add_argument('--output_annotation_dir', type=str, required=True, help='Directory to save annotations')

    args = parser.parse_args()

    prepare_detection_dataset(args.video_path, args.annotation_path, args.output_image_dir, args.output_flow_dir, args.output_annotation_dir)

