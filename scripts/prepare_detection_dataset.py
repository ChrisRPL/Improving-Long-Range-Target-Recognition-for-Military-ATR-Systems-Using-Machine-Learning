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

def prepare_optical_flow(data_root: Path):
    """Prepare optical flow for all dataset splits."""
    data_root = Path(data_root)
    
    for split in ['train', 'valid', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Setup paths
        images_dir = data_root / split / 'images'
        flow_dir = data_root / split / 'flow'
        flow_dir.mkdir(exist_ok=True, parents=True)
        
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            continue
        
        # Get sorted image paths
        image_paths = sorted(list(images_dir.glob('*.jpg')))
        if not image_paths:
            image_paths = sorted(list(images_dir.glob('*.png')))
        
        if not image_paths:
            print(f"No images found in {images_dir}")
            continue
            
        print(f"Found {len(image_paths)} images")
        
        # Group images by sequence
        sequences = {}
        for img_path in image_paths:
            # Adjust this based on your image naming convention
            # Example: if images are named like "video_001_frame_001.jpg"
            seq_id = '_'.join(img_path.stem.split('_')[:2])
            if seq_id not in sequences:
                sequences[seq_id] = []
            sequences[seq_id].append(img_path)
        
        print(f"Found {len(sequences)} sequences")
        
        # Process each sequence
        total_flows = 0
        for seq_id, seq_paths in tqdm(sequences.items(), desc=f"Processing {split} sequences"):
            flows = process_sequence(sorted(seq_paths))
            
            # Save flows
            for img_path, flow in flows:
                flow_path = flow_dir / f"{img_path.stem}_flow.npy"
                np.save(str(flow_path), flow)
                total_flows += 1
        
        print(f"Generated {total_flows} flow files for {split} split")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    args = parser.parse_args()
    
    # Load data.yaml
    import yaml
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get dataset root from training path
    train_path = Path(data_config['train'])
    dataset_root = train_path.parent.parent
    
    print(f"Dataset root: {dataset_root}")
    prepare_optical_flow(dataset_root)
