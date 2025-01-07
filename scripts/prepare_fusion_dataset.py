import os
import cv2
import argparse
import sys
import json
from pathlib import Path
from utils.video_processing import extract_frames, align_frames, compute_optical_flow, wavelet_fusion, save_fused_frames

sys.path.append(str(Path(__file__).parent.parent))

def prepare_fusion_dataset(visible_path, mwir_path, output_path, matching_dict_path, batch_size=10):
    # Get sorted filenames
    with open(matching_dict_path) as json_file:
       data = json.load(json_file)
       
    visible_files = list(data.keys())
    mwir_files = list(data.values())
    
    
    for i in range(0, len(visible_files), batch_size):
        batch_visible_files = visible_files[i:i + batch_size]
        batch_mwir_files = mwir_files[i:i + batch_size]
        
        # Load batch
        visible_frames = []
        mwir_frames = []
        for v_file, m_file in zip(batch_visible_files, batch_mwir_files):
            v_frame = cv2.imread(os.path.join(visible_path, v_file))
            m_frame = cv2.imread(os.path.join(mwir_path, m_file))
            visible_frames.append(v_frame)
            mwir_frames.append(m_frame)
        
        # Process batch
        target_size = (visible_frames[0].shape[1], visible_frames[0].shape[0])
        for v_frame, m_frame, filename in zip(visible_frames, mwir_frames, batch_visible_files):
            m_frame_resized = cv2.resize(m_frame, target_size)
            fused_frame = wavelet_fusion(v_frame, m_frame_resized)
            cv2.imwrite(os.path.join(output_path, filename), fused_frame)
        
        # Clear memory
        visible_frames = []
        mwir_frames = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Fusion Dataset")
    parser.add_argument('--visible_video_path', type=str, required=True, help='Path to the visible spectrum video')
    parser.add_argument('--mwir_video_path', type=str, required=True, help='Path to the MWIR video')
    parser.add_argument('--output_video_path', type=str, required=True, help='Path to save the fused video')
    parser.add_argument('--matching_dict_path', type=str, required=True, help='Path to json of rgb-thermal pairs')

    args = parser.parse_args()

    prepare_fusion_dataset(args.visible_video_path, args.mwir_video_path, args.output_video_path, args.matching_dict_path)

