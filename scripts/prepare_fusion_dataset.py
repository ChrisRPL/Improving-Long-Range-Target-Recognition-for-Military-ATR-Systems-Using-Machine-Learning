import os
import cv2
import argparse
import sys
from pathlib import Path
from utils.video_processing import extract_frames, align_frames, compute_optical_flow, wavelet_fusion, save_fused_frames

sys.path.append(str(Path(__file__).parent.parent))

def prepare_fusion_dataset(visible_video_path, mwir_video_path, output_video_path):
    # Extract frames and filenames
    visible_frames, visible_filenames = extract_frames(visible_video_path)
    mwir_frames, mwir_filenames = extract_frames(mwir_video_path)
    
    print(f"Visible frames: {len(visible_frames)}, MWIR frames: {len(mwir_frames)}")
    
    # Ensure both lists have same length
    min_frames = min(len(visible_frames), len(mwir_frames))
    visible_frames = visible_frames[:min_frames]
    mwir_frames = mwir_frames[:min_frames]
    filenames = visible_filenames[:min_frames]
    
    # Get target size from visible frame
    target_size = (visible_frames[0].shape[1], visible_frames[0].shape[0])
    
    # Process frames
    fused_frames = []
    for v_frame, m_frame in zip(visible_frames, mwir_frames):
        # Resize MWIR to match visible
        m_frame_resized = cv2.resize(m_frame, (v_frame.shape[1], v_frame.shape[0]))
        # Fuse frames
        fused_frame = wavelet_fusion(v_frame, m_frame_resized)
        fused_frames.append(fused_frame)
    
    save_fused_frames(fused_frames, output_video_path, filenames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Fusion Dataset")
    parser.add_argument('--visible_video_path', type=str, required=True, help='Path to the visible spectrum video')
    parser.add_argument('--mwir_video_path', type=str, required=True, help='Path to the MWIR video')
    parser.add_argument('--output_video_path', type=str, required=True, help='Path to save the fused video')

    args = parser.parse_args()

    prepare_fusion_dataset(args.visible_video_path, args.mwir_video_path, args.output_video_path)

