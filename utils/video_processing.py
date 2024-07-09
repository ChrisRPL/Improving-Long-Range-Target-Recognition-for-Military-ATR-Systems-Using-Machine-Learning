import cv2
import numpy as np
import pywt
from skimage.feature import match_descriptors, ORB
from skimage.transform import ProjectiveTransform, warp
from skimage.measure import ransac

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames

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
        # Resize MWIR frame to the target size (same as visible frame)
        mwir_resized = cv2.resize(mwir_frames[i], target_size)
        
        keypoints1, keypoints2, matches = detect_and_match_features(visible_frames[i], mwir_resized)

        src = keypoints2[matches[:, 1]][:, ::-1]
        dst = keypoints1[matches[:, 0]][:, ::-1]

        model_robust, inliers = ransac((src, dst), ProjectiveTransform, min_samples=4,
                                       residual_threshold=2, max_trials=300)
        warped = warp(mwir_resized, model_robust.inverse, output_shape=visible_frames[i].shape)
        aligned_mwir_frames.append(warped)
    return aligned_mwir_frames

def compute_optical_flow(visible_frames, aligned_mwir_frames):
    flow_frames = []
    for v_frame, m_frame in zip(visible_frames, aligned_mwir_frames):
        flow = cv2.calcOpticalFlowFarneback(m_frame, v_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        h, w = flow.shape[:2]
        flow_map = np.zeros_like(flow, dtype=np.float32)
        flow_map[:, :, 0] = np.repeat(np.arange(w), h).reshape(w, h).T + flow[:, :, 0]
        flow_map[:, :, 1] = np.tile(np.arange(h), w).reshape(h, w) + flow[:, :, 1]
        
        remapped_frame = cv2.remap(m_frame, flow_map, None, cv2.INTER_LINEAR)
        flow_frames.append(remapped_frame)
    return flow_frames

def wavelet_fusion(visible_frame, mwir_frame):
    coeffs_visible = pywt.dwt2(visible_frame, 'db1')
    coeffs_mwir = pywt.dwt2(mwir_frame, 'db1')

    fused_coeffs = (
        (coeffs_visible[0] + coeffs_mwir[0]) / 2,
        tuple((cv2.add(c_v, c_m) / 2) for c_v, c_m in zip(coeffs_visible[1], coeffs_mwir[1]))
    )

    fused_frame = pywt.idwt2(fused_coeffs, 'db1')
    return fused_frame

def save_fused_video(fused_frames, output_path, fps=30):
    height, width = fused_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    for frame in fused_frames:
        out.write(cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
    out.release()

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
