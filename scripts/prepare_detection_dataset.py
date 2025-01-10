import cv2
import numpy as np
from pathlib import Path
import concurrent.futures

def calculate_optical_flow(prev_frame, curr_frame):
    """Calculate optical flow using Farneback algorithm."""
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

def process_sequence(image_paths):
    """Process a sequence of images and generate optical flow."""
    flows = []
    prev_frame = None
    
    for img_path in image_paths:
        curr_frame = cv2.imread(str(img_path))
        if prev_frame is not None:
            flow = calculate_optical_flow(prev_frame, curr_frame)
            flows.append((img_path, flow))
        prev_frame = curr_frame
    
    return flows

def prepare_optical_flow(data_root: Path):
    """Prepare optical flow for all dataset splits."""
    for split in ['train', 'valid', 'test']:
        print(f"Processing {split} split...")
        
        # Setup paths
        images_dir = data_root / split / 'images'
        flow_dir = data_root / split / 'flow'
        flow_dir.mkdir(exist_ok=True)
        
        # Get sorted image paths
        image_paths = sorted(list(images_dir.glob('*.jpg')))
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Group images by video sequence
            sequences = {}
            for img_path in image_paths:
                seq_id = img_path.stem.split('_')[0]  # Adjust based on your naming convention
                if seq_id not in sequences:
                    sequences[seq_id] = []
                sequences[seq_id].append(img_path)
            
            # Process each sequence
            futures = []
            for seq_images in sequences.values():
                futures.append(executor.submit(process_sequence, seq_images))
            
            # Save flows
            for future in concurrent.futures.as_completed(futures):
                for img_path, flow in future.result():
                    flow_path = flow_dir / f"{img_path.stem}_flow.npy"
                    np.save(str(flow_path), flow)
        
        print(f"Completed {split} split")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    args = parser.parse_args()
    
    prepare_optical_flow(Path(args.data_root))
