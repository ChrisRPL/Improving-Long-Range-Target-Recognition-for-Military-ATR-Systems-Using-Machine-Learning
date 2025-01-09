import os
import cv2
import torch
import numpy as np
import json
from torchvision import transforms
from models.dsgan import DSGenerator
from pathlib import Path
import argparse
from PIL import Image
import sys

sys.path.append(str(Path(__file__).parent.parent))

def calculate_optical_flow(prev_frame, curr_frame):
    """Calculate optical flow between two frames using Farneback method."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, 
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

def load_generator(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = DSGenerator().to(device)
    
    try:
        # Load with warning about pickle security
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        if isinstance(checkpoint, dict):
            if 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
            elif 'state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['state_dict'])
            else:
                print("Checkpoint structure:", checkpoint.keys())
                raise ValueError("Unexpected checkpoint structure")
        else:
            generator.load_state_dict(checkpoint)
            
        print("Successfully loaded generator model")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
        
    generator.eval()
    return generator, device

def load_coco_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

class SmallObjectInpainter:
    def __init__(self, generator, device, min_size=96):
        self.generator = generator
        self.device = device
        self.min_size = min_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def process_image(self, frame, large_objects, image_info):
        """Process a single image."""
        height, width = frame.shape[:2]
        new_annotations = []
        processed = False
    
        print(f"\nProcessing image of size {width}x{height}")
        print(f"Number of large objects to process: {len(large_objects)}")
    
        for ann in large_objects:
            bbox = ann['bbox']
            x, y, w, h = [int(v) for v in bbox]
        
            print(f"Processing object: x={x}, y={y}, w={w}, h={h}")
        
            # Extract large object
            large_object = frame[y:y+h, x:x+w]
            if large_object.size == 0:
                print("Skipping: empty crop")
                continue
            
            # Generate smaller version
            print("Generating smaller version...")
            small_object = self.generate_small_object(large_object)
        
            # Calculate new size and position
            new_w = int(w * 0.5)
            new_h = int(h * 0.5)
        
            offset_x = int((np.random.random() - 0.5) * w * 0.2)
            offset_y = int((np.random.random() - 0.5) * h * 0.2)
        
            new_x = max(0, x + offset_x)
            new_y = max(0, y + offset_y)
            new_x = min(new_x, width - new_w)
            new_y = min(new_y, height - new_h)
        
            print(f"New position: x={new_x}, y={new_y}, w={new_w}, h={new_h}")
        
            # Resize and place small object
            small_object_resized = cv2.resize(small_object, (new_w, new_h))
            frame[new_y:new_y+new_h, new_x:new_x+new_w] = small_object_resized
            processed = True
        
            # Create new annotation
            new_ann = ann.copy()
            new_ann['id'] = len(new_annotations) + max([a['id'] for a in large_objects]) + 1
            new_ann['bbox'] = [float(new_x), float(new_y), float(new_w), float(new_h)]
            new_ann['area'] = float(new_w * new_h)
            new_annotations.append(new_ann)
            print(f"Added new annotation with id {new_ann['id']}")
    
        return frame if processed else None, new_annotations

    def generate_small_object(self, large_object):
        if len(large_object.shape) == 2:
            large_object = cv2.cvtColor(large_object, cv2.COLOR_GRAY2RGB)
        
        large_object = Image.fromarray(large_object)
        large_object_tensor = self.transform(large_object).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            small_object = self.generator(large_object_tensor)
            small_object = small_object.cpu().squeeze(0).permute(1, 2, 0)
            small_object = ((small_object + 1) * 127.5).numpy().astype(np.uint8)
        
        return small_object

    def track_object_with_flow(self, prev_frame, curr_frame, prev_bbox):
        """Track object using optical flow."""
        flow = calculate_optical_flow(prev_frame, curr_frame)
        x, y, w, h = prev_bbox
        
        # Calculate average flow in the object region
        roi_flow = flow[y:y+h, x:x+w]
        mean_flow = np.mean(roi_flow, axis=(0, 1))
        
        # Update bbox based on flow
        new_x = int(x + mean_flow[0])
        new_y = int(y + mean_flow[1])
        
        # Ensure bbox stays within image bounds
        h, w = curr_frame.shape[:2]
        new_x = max(0, min(new_x, w - prev_bbox[2]))
        new_y = max(0, min(new_y, h - prev_bbox[3]))
        
        return [new_x, new_y, prev_bbox[2], prev_bbox[3]]

    def process_video_sequence(self, frames, annotations, image_info):
        """Process a sequence of frames with optical flow tracking."""
        processed_frames = []
        new_annotations = []
        height, width = frames[0].shape[:2]
        
        # Filter large objects from first frame
        large_objects = [
            ann for ann in annotations 
            if ann['bbox'][2] * ann['bbox'][3] >= self.min_size * self.min_size
        ]
        
        # Process each large object
        for ann in large_objects:
            curr_bbox = ann['bbox']
            x, y, w, h = [int(v) for v in curr_bbox]
            
            # Generate small object from large object
            large_object = frames[0][y:y+h, x:x+w]
            if large_object.size == 0:
                continue
                
            small_object = self.generate_small_object(large_object)
            new_w = int(w * 0.5)
            new_h = int(h * 0.5)
            small_object_resized = cv2.resize(small_object, (new_w, new_h))
            
            # Track and place object in each frame
            curr_small_bbox = [x, y, new_w, new_h]
            for frame_idx, frame in enumerate(frames):
                if frame_idx == 0:
                    # Random initial placement for first frame
                    offset_x = int((np.random.random() - 0.5) * w * 0.2)
                    offset_y = int((np.random.random() - 0.5) * h * 0.2)
                    new_x = max(0, min(x + offset_x, width - new_w))
                    new_y = max(0, min(y + offset_y, height - new_h))
                else:
                    # Use optical flow for subsequent frames
                    curr_small_bbox = self.track_object_with_flow(
                        frames[frame_idx-1], frame, curr_small_bbox
                    )
                    new_x, new_y = curr_small_bbox[0], curr_small_bbox[1]
                
                # Place small object
                frame[new_y:new_y+new_h, new_x:new_x+new_w] = small_object_resized
                
                # Create annotation for current frame
                new_ann = ann.copy()
                new_ann['id'] = len(new_annotations) + max([a['id'] for a in annotations]) + 1
                new_ann['image_id'] = image_info['id'] + frame_idx
                new_ann['bbox'] = [float(new_x), float(new_y), float(new_w), float(new_h)]
                new_ann['area'] = float(new_w * new_h)
                new_annotations.append(new_ann)
                
            processed_frames.append(frame)
        
        return processed_frames, new_annotations

def inpaint_dataset_with_gan(args):
    generator, device = load_generator(args.model_path)
    inpainter = SmallObjectInpainter(generator, device)
    
    print("Loading COCO annotations...")
    coco_data = load_coco_annotations(args.annotation_path)
    print(f"Loaded {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_annotation_dir = Path(args.output_annotation_path).parent
    output_annotation_dir.mkdir(parents=True, exist_ok=True)
    
    new_coco_data = {
        'images': coco_data['images'].copy(),
        'categories': coco_data['categories'],
        'annotations': coco_data['annotations'].copy()
    }
    
    # Counter for progress tracking
    total_images = len(coco_data['images'])
    processed_count = 0
    large_objects_found = 0
    total_new_annotations = 0
    
    print("\nProcessing images...")
    for img_info in coco_data['images']:
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processing image {processed_count}/{total_images}")
            
        try:
            image_path = Path(args.image_dir) / img_info['file_name']
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
                
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Warning: Could not read image: {image_path}")
                continue
            
            # Get annotations for this image
            image_annotations = [
                ann for ann in coco_data['annotations'] 
                if ann['image_id'] == img_info['id']
            ]
            
            # Filter large objects
            large_objects = [
                ann for ann in image_annotations 
                if ann['bbox'][2] * ann['bbox'][3] >= 96 * 96  # min_size check
            ]
            
            if large_objects:
                large_objects_found += 1
                print(f"\nFound {len(large_objects)} large objects in {img_info['file_name']}")
                
                # Process image
                processed_frame, new_anns = inpainter.process_image(
                    frame, 
                    large_objects,
                    img_info
                )
                
                if processed_frame is not None and len(new_anns) > 0:
                    # Save processed image
                    output_path = output_dir / img_info['file_name']
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save the image
                    success = cv2.imwrite(str(output_path), processed_frame)
                    if success:
                        print(f"Saved processed image to {output_path}")
                        new_coco_data['annotations'].extend(new_anns)
                        total_new_annotations += len(new_anns)
                    else:
                        print(f"Failed to save image to {output_path}")
            
        except Exception as e:
            print(f"Error processing image {img_info['file_name']}: {str(e)}")
            continue
    
    print("\nProcessing summary:")
    print(f"Total images processed: {processed_count}")
    print(f"Images with large objects: {large_objects_found}")
    print(f"New annotations created: {total_new_annotations}")
    
    # Save updated annotations
    print("\nSaving annotations...")
    output_annotation_path = Path(args.output_annotation_path)
    if output_annotation_path.suffix != '.json':
        output_annotation_path = output_annotation_path / 'annotations.json'
    
    with open(output_annotation_path, 'w') as f:
        json.dump(new_coco_data, f)
    
    print(f"Annotations saved to {output_annotation_path}")
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpaint Dataset with DS-GAN")
    parser.add_argument('--image_dir', type=str, required=True, 
                      help='Directory containing input images')
    parser.add_argument('--annotation_path', type=str, required=True, 
                      help='Path to COCO format annotation file')
    parser.add_argument('--output_dir', type=str, required=True, 
                      help='Directory to save processed images')
    parser.add_argument('--output_annotation_path', type=str, required=True, 
                      help='Path to save updated COCO annotations')
    parser.add_argument('--model_path', type=str, required=True, 
                      help='Path to trained DS-GAN generator model')
    
    args = parser.parse_args()
    inpaint_dataset_with_gan(args)
