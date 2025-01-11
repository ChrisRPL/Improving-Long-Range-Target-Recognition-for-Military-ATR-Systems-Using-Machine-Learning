import cv2
import numpy as np
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

def calculate_optical_flow(prev_frame, curr_frame):
    """Calculate optical flow using Farneback algorithm."""
    try:
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow
    except Exception as e:
        print(f"Error calculating optical flow: {e}")
        return None

def process_sequence(sequence_paths):
    """Process a sequence of images and generate optical flow."""
    flows = []
    
    if len(sequence_paths) < 2:
        return flows
    
    for i in range(len(sequence_paths) - 1):
        try:
            # Read consecutive frames
            prev_frame = cv2.imread(str(sequence_paths[i]))
            curr_frame = cv2.imread(str(sequence_paths[i+1]))
            
            if prev_frame is None or curr_frame is None:
                print(f"Could not read frames: {sequence_paths[i]} or {sequence_paths[i+1]}")
                continue
                
            # Calculate flow
            flow = calculate_optical_flow(prev_frame, curr_frame)
            if flow is not None:
                flows.append((sequence_paths[i+1], flow))
            
        except Exception as e:
            print(f"Error processing sequence {sequence_paths[i]}: {e}")
            continue
    
    return flows

def prepare_optical_flow(dataset_root: Path):
    """Prepare optical flow for all dataset splits."""
    dataset_root = Path(dataset_root)
    print(f"Processing dataset at: {dataset_root}")
    
    for split in ['train', 'valid', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Setup paths
        images_dir = dataset_root / split / 'images'
        flow_dir = dataset_root / split / 'flow'
        
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            continue
            
        flow_dir.mkdir(exist_ok=True, parents=True)
        
        # Get sorted image paths
        image_paths = sorted(list(images_dir.glob('*.jpg')))
        if not image_paths:
            image_paths = sorted(list(images_dir.glob('*.png')))
        
        if not image_paths:
            print(f"No images found in {images_dir}")
            continue
            
        print(f"Found {len(image_paths)} images")
        
        # Process all images sequentially
        flows = process_sequence(image_paths)
        
        # Save flows
        print(f"Saving {len(flows)} flow files...")
        for img_path, flow in tqdm(flows, desc=f"Saving {split} flows"):
            flow_path = flow_dir / f"{img_path.stem}_flow.npy"
            np.save(str(flow_path), flow)
        
        print(f"Generated {len(flows)} flow files for {split} split")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, 
                      help='Root directory of dataset containing train/valid/test folders')
    args = parser.parse_args()
    
    prepare_optical_flow(args.dataset_root)
