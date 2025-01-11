import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc  # Garbage collector

def calculate_optical_flow(prev_frame, curr_frame):
    """Calculate optical flow using Farneback algorithm."""
    try:
        # Ensure frames are the same size
        if prev_frame.shape != curr_frame.shape:
            curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
        
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,  # No initial flow
            0.5,   # Pyramid scale
            3,     # Pyramid levels
            15,    # Window size
            3,     # Iterations
            5,     # Poly neighborhood
            1.2,   # Poly sigma
            0      # Flags
        )
        
        # Clear memory
        del prev_gray, curr_gray
        gc.collect()
        
        return flow
    except Exception as e:
        print(f"Error calculating optical flow: {e}")
        return None

def process_batch(image_paths, batch_size=100):
    """Process a batch of images."""
    flows = []
    
    for i in range(0, len(image_paths) - 1, batch_size):
        batch_paths = image_paths[i:i + batch_size + 1]  # +1 to ensure overlap between batches
        
        for j in range(len(batch_paths) - 1):
            try:
                # Read consecutive frames
                prev_frame = cv2.imread(str(batch_paths[j]))
                curr_frame = cv2.imread(str(batch_paths[j + 1]))
                
                if prev_frame is None or curr_frame is None:
                    print(f"Could not read frames: {batch_paths[j]} or {batch_paths[j + 1]}")
                    continue
                
                # Calculate flow
                flow = calculate_optical_flow(prev_frame, curr_frame)
                if flow is not None:
                    flows.append((batch_paths[j + 1], flow))
                
                # Clear memory
                del prev_frame, curr_frame
                gc.collect()
                
            except Exception as e:
                print(f"Error processing pair {batch_paths[j]} -> {batch_paths[j + 1]}: {e}")
                continue
            
        # Clear memory after each batch
        gc.collect()
    
    return flows

def prepare_optical_flow(dataset_root: Path, batch_size: int = 100):
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
        
        # Process images in batches
        total_processed = 0
        batch_idx = 0
        while batch_idx < len(image_paths):
            print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
            
            # Process batch
            batch_end = min(batch_idx + batch_size, len(image_paths))
            batch_paths = image_paths[batch_idx:batch_end]
            flows = process_batch([batch_paths[0]] + batch_paths)  # Include previous frame for first image
            
            # Save flows
            print(f"Saving {len(flows)} flow files...")
            for img_path, flow in tqdm(flows, desc=f"Saving flows"):
                flow_path = flow_dir / f"{img_path.stem}_flow.npy"
                np.save(str(flow_path), flow)
                total_processed += 1
            
            # Update batch index
            batch_idx = batch_end
            
            # Clear memory
            gc.collect()
        
        print(f"Generated {total_processed} flow files for {split} split")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, 
                      help='Root directory of dataset containing train/valid/test folders')
    parser.add_argument('--batch_size', type=int, default=100,
                      help='Number of images to process in each batch')
    args = parser.parse_args()
    
    prepare_optical_flow(args.dataset_root, args.batch_size)
