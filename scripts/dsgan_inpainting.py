import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from models.dsgan import DSGenerator
from utils.annotations import load_yolo_annotations, save_annotations
from utils.video_processing import extract_frames, calculate_optical_flow
import argparse

# Load pre-trained models
def load_generator(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = DSGenerator().to(device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    return generator

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def generate_smaller_object(generator, large_object):
    large_object_tensor = transform(large_object).unsqueeze(0).to(device)
    with torch.no_grad():
        smaller_object = generator(large_object_tensor).cpu().numpy().squeeze().transpose(1, 2, 0)
    smaller_object = (smaller_object * 255).astype(np.uint8)
    return smaller_object

def inpaint_video_with_gan(video_path, annotation_path, output_video_path, generator, output_annotation_path):
    frames = extract_frames(video_path)
    annotations = load_yolo_annotations(annotation_path)
    inpainted_frames = []
    new_annotations = []

    for frame_number, (class_id, x_center, y_center, width, height) in enumerate(annotations):
        frame = frames[frame_number]
        h, w = frame.shape[:2]
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        large_object = frame[y1:y2, x1:x2]
        smaller_object = generate_smaller_object(generator, large_object)
        
        smaller_x_center = x_center + (np.random.rand() - 0.5) * 0.1  # Random shift in center
        smaller_y_center = y_center + (np.random.rand() - 0.5) * 0.1
        smaller_width = width * 0.5  # Smaller object is half the width
        smaller_height = height * 0.5
        
        sx1 = int((smaller_x_center - smaller_width / 2) * w)
        sy1 = int((smaller_y_center - smaller_height / 2) * h)
        sx2 = int((smaller_x_center + smaller_width / 2) * w)
        sy2 = int((smaller_y_center + smaller_height / 2) * h)
        
        frame[sy1:sy2, sx1:sx2] = cv2.resize(smaller_object, (sx2 - sx1, sy2 - sy1))

        new_annotation = (class_id, smaller_x_center, smaller_y_center, smaller_width, smaller_height)
        new_annotations.append(new_annotation)

        inpainted_frames.append(frame)
    
    # Optical flow to track object motion and inpaint in subsequent frames
    for i in range(1, len(inpainted_frames)):
        prev_frame = inpainted_frames[i - 1]
        current_frame = inpainted_frames[i]
        flow = calculate_optical_flow(prev_frame, current_frame)
        h, w = flow.shape[:2]
        flow_map = np.zeros_like(flow, dtype=np.float32)
        flow_map[:, :, 0] = np.repeat(np.arange(w), h).reshape(w, h).T + flow[:, :, 0]
        flow_map[:, :, 1] = np.tile(np.arange(h), w).reshape(h, w) + flow[:, :, 1]
        
        remapped_object = cv2.remap(smaller_object, flow_map, None, cv2.INTER_LINEAR)
        current_frame[sy1:sy2, sx1:sx2] = remapped_object

    save_annotations(new_annotations, output_annotation_path)
    save_fused_video(inpainted_frames, output_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpaint Video with DS-GAN")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to the annotation file')
    parser.add_argument('--output_video_path', type=str, required=True, help='Path to save the output video')
    parser.add_argument('--output_annotation_path', type=str, required=True, help='Path to save the output annotation file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained DS-GAN generator model')

    args = parser.parse_args()

    generator = load_generator(args.model_path)
    inpaint_video_with_gan(args.video_path, args.annotation_path, args.output_video_path, generator, args.output_annotation_path)

