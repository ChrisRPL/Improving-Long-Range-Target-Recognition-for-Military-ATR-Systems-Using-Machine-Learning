import os
import cv2
import argparse
from utils.video_processing import extract_frames, align_frames, compute_optical_flow, wavelet_fusion, save_fused_video

def prepare_fusion_dataset(visible_video_path, mwir_video_path, output_video_path):
    visible_frames = extract_frames(visible_video_path)
    mwir_frames = extract_frames(mwir_video_path)

    min_frames = min(len(visible_frames), len(mwir_frames))
    visible_frames = visible_frames[:min_frames]
    mwir_frames = mwir_frames[:min_frames]

    target_size = (visible_frames[0].shape[1], visible_frames[0].shape[0])

    aligned_mwir_frames = align_frames(visible_frames, mwir_frames, target_size)
    refined_mwir_frames = compute_optical_flow(visible_frames, aligned_mwir_frames)

    fused_frames = []
    for v_frame, m_frame in zip(visible_frames, refined_mwir_frames):
        fused_frame = wavelet_fusion(v_frame, m_frame)
        fused_frames.append(fused_frame)

    save_fused_video(fused_frames, output_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Fusion Dataset")
    parser.add_argument('--visible_video_path', type=str, required=True, help='Path to the visible spectrum video')
    parser.add_argument('--mwir_video_path', type=str, required=True, help='Path to the MWIR video')
    parser.add_argument('--output_video_path', type=str, required=True, help='Path to save the fused video')

    args = parser.parse_args()

    prepare_fusion_dataset(args.visible_video_path, args.mwir_video_path, args.output_video_path)

