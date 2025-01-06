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
    for i in range(len(visible_frames)):
        mwir_resized = cv2.resize(mwir_frames[i], target_size)
        
        visible_gray = cv2.cvtColor(visible_frames[i], cv2.COLOR_BGR2GRAY)
        mwir_gray = cv2.cvtColor(mwir_resized, cv2.COLOR_BGR2GRAY)
        
        keypoints1, keypoints2, matches = detect_and_match_features(visible_gray, mwir_gray)

        src = keypoints2[matches[:, 1]][:, ::-1]
        dst = keypoints1[matches[:, 0]][:, ::-1]

        model_robust, inliers = ransac((src, dst), ProjectiveTransform, min_samples=4,
                                     residual_threshold=2, max_trials=300)
        warped = warp(mwir_resized, model_robust.inverse, output_shape=visible_frames[i].shape)
        # Convert to uint8
        warped = (warped * 255).astype(np.uint8)
        aligned_mwir_frames.append(warped)
    return aligned_mwir_frames

def compute_optical_flow(visible_frames, aligned_mwir_frames):
    flow_frames = []
    for v_frame, m_frame in zip(visible_frames, aligned_mwir_frames):
        # Ensure frames are uint8
        v_frame = v_frame.astype(np.uint8)
        m_frame = m_frame.astype(np.uint8)
        
        v_gray = cv2.cvtColor(v_frame, cv2.COLOR_BGR2GRAY)
        m_gray = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(m_gray, v_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        h, w = flow.shape[:2]
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        flow_map[:, :, 0] = np.repeat(np.arange(w), h).reshape(w, h).T + flow[:, :, 0]
        flow_map[:, :, 1] = np.tile(np.arange(h), w).reshape(h, w) + flow[:, :, 1]
        
        remapped_frame = cv2.remap(m_frame, flow_map[:, :, 0], flow_map[:, :, 1], cv2.INTER_LINEAR)
        flow_frames.append(remapped_frame)
    return flow_frames

def wavelet_fusion(visible_frame, mwir_frame):
   fused_frame = np.zeros_like(visible_frame)
   wavelet = 'sym4'
   levels = 3  # Multi-level decomposition
   
   for channel in range(3):
       # Multi-level wavelet decomposition
       coeffs_visible = pywt.wavedec2(visible_frame[:,:,channel], wavelet, level=levels)
       coeffs_mwir = pywt.wavedec2(mwir_frame[:,:,channel], wavelet, level=levels)
       
       # Fusion rules for each level
       fused_coeffs = []
       for level in range(levels + 1):
           if level == 0:  # Approximation coefficients
               # Calculate local entropy-based weights
               entropy_visible = entropy(coeffs_visible[level])
               entropy_mwir = entropy(coeffs_mwir[level])
               
               w_visible = entropy_visible / (entropy_visible + entropy_mwir)
               w_mwir = entropy_mwir / (entropy_visible + entropy_mwir)
               
               fused_coeffs.append(w_visible * coeffs_visible[level] + 
                                 w_mwir * coeffs_mwir[level])
           else:
               # Detail coefficients - preserve edges
               detail_coeffs = []
               for v_coeff, m_coeff in zip(coeffs_visible[level], coeffs_mwir[level]):
                   # Edge strength
                   edge_v = np.abs(v_coeff)
                   edge_m = np.abs(m_coeff)
                   
                   mask = edge_v > edge_m
                   fused_detail = np.where(mask, v_coeff, m_coeff)
                   detail_coeffs.append(fused_detail)
               
               fused_coeffs.append(tuple(detail_coeffs))
       
       # Inverse wavelet transform
       channel_fused = pywt.waverec2(fused_coeffs, wavelet)
       fused_frame[:,:,channel] = channel_fused

   # Edge-preserving filtering
   fused_frame = cv2.edgePreservingFilter(fused_frame, flags=1, sigma_s=60, sigma_r=0.4)
   
   # Adaptive contrast enhancement
   lab = cv2.cvtColor(fused_frame.astype(np.uint8), cv2.COLOR_BGR2LAB)
   l, a, b = cv2.split(lab)
   clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
   l = clahe.apply(l)
   lab = cv2.merge((l,a,b))
   fused_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
   
   return fused_frame

def entropy(img):
   """Calculate local entropy"""
   hist = np.histogram(img, bins=256, density=True)[0]
   hist = hist[hist > 0]
   return -np.sum(hist * np.log2(hist))

def save_fused_frames(fused_frames, output_folder, original_filenames):
   os.makedirs(output_folder, exist_ok=True)
   for frame, filename in zip(fused_frames, original_filenames):
       normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
       output_path = os.path.join(output_folder, filename)
       cv2.imwrite(output_path, normalized)

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
