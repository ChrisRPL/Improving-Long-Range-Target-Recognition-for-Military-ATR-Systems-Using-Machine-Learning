import cv2
import numpy as np
import pywt
import os
from skimage.feature import match_descriptors, ORB
from skimage.transform import ProjectiveTransform, warp
from skimage.measure import ransac

def extract_frames(folder_path):
   frames = []
   filenames = []
   for filename in sorted(os.listdir(folder_path)):
       if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
           image_path = os.path.join(folder_path, filename)
           frame = cv2.imread(image_path)
           frames.append(frame)
           filenames.append(filename)
   return frames, filenames

def detect_and_match_features(img1, img2):
    orb = ORB(n_keypoints=500)
    orb.detect_and_extract(img1)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    orb.detect_and_extract(img2)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors

    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    return keypoints1, keypoints2, matches

def align_frames(visible_frames, mwir_frames, target_size):
    aligned_mwir_frames = []
    for mwir_frame in mwir_frames:
        aligned_mwir_frames.append(cv2.resize(mwir_frame, target_size))
    return aligned_mwir_frames

def compute_optical_flow(visible_frames, aligned_mwir_frames):
    return aligned_mwir_frames  # Skip optical flow for now

def wavelet_fusion(visible_frame, mwir_frame):
    # Ensure both frames are uint8
    visible_frame = np.clip(visible_frame, 0, 255).astype(np.uint8)
    mwir_frame = np.clip(mwir_frame, 0, 255).astype(np.uint8)
    
    # Convert to grayscale
    mwir_gray = cv2.cvtColor(mwir_frame, cv2.COLOR_BGR2GRAY)
    mwir_colored = cv2.cvtColor(mwir_gray, cv2.COLOR_GRAY2BGR)
    
    # Simple alpha blending
    alpha = 0.6
    beta = 0.4
    fused = cv2.addWeighted(visible_frame, alpha, mwir_colored, beta, 0)
    
    return fused

def save_fused_frames(fused_frames, output_folder, original_filenames):
    os.makedirs(output_folder, exist_ok=True)
    for frame, filename in zip(fused_frames, original_filenames):
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, frame)

def fuse_videos(visible_video_path, mwir_video_path, output_video_path):
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
