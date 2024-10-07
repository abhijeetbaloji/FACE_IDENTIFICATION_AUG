import cv2
import numpy as np
from src.webcam_constants import (
    BLUR_KERNEL_SIZE,
    SUNGLASSES_IMAGE_PATH,
    MUSTACHE_IMAGE_PATH,
    HAIRSTYLE_IMAGE_PATH,
    FACE_MASK_IMAGE_PATH
)

def apply_blur_filter(frame, landmarks):
    if not landmarks:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for face_landmarks in landmarks:
        hull = cv2.convexHull(np.array(face_landmarks))
        cv2.fillConvexPoly(mask, hull, 255)
    blurred_frame = cv2.GaussianBlur(frame, BLUR_KERNEL_SIZE, 0)
    frame = np.where(mask[:, :, np.newaxis] == 255, blurred_frame, frame)
    return frame

def apply_sunglasses_filter(frame, landmarks):
    if not landmarks:
        return frame
    sunglasses = cv2.imread(SUNGLASSES_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        print(f"Error: Unable to load sunglasses image from {SUNGLASSES_IMAGE_PATH}")
        return frame
    for face_landmarks in landmarks:
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        eye_width = int(np.linalg.norm(np.array(right_eye) - np.array(left_eye)))
        sunglasses_width = int(eye_width * 2.2)
        aspect_ratio = sunglasses.shape[0] / sunglasses.shape[1]
        sunglasses_height = int(sunglasses_width * aspect_ratio)
        resized_sunglasses = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height), interpolation=cv2.INTER_AREA)
        eye_delta_x = right_eye[0] - left_eye[0]
        eye_delta_y = right_eye[1] - left_eye[1]
        angle = -np.degrees(np.arctan2(eye_delta_y, eye_delta_x))
        M = cv2.getRotationMatrix2D((sunglasses_width // 2, sunglasses_height // 2), angle, 1.0)
        rotated_sunglasses = cv2.warpAffine(resized_sunglasses, M, (sunglasses_width, sunglasses_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        center = np.mean([left_eye, right_eye], axis=0).astype(int)
        top_left = (int(center[0] - sunglasses_width / 2), int(center[1] - sunglasses_height / 2))
        top_left_y = max(0, top_left[1])
        bottom_right_y = min(frame.shape[0], top_left[1] + sunglasses_height)
        top_left_x = max(0, top_left[0])
        bottom_right_x = min(frame.shape[1], top_left[0] + sunglasses_width)
        roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        sunglasses_roi = rotated_sunglasses[top_left_y - top_left[1]:bottom_right_y - top_left[1], top_left_x - top_left[0]:bottom_right_x - top_left[0]]
        for i in range(sunglasses_roi.shape[0]):
            for j in range(sunglasses_roi.shape[1]):
                if sunglasses_roi[i, j, 3] > 0:
                    roi[i, j] = sunglasses_roi[i, j, :3]
    return frame

def apply_mustache_filter(frame, landmarks):
    if not landmarks:
        return frame
    mustache = cv2.imread(MUSTACHE_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if mustache is None:
        print(f"Error: Unable to load mustache image from {MUSTACHE_IMAGE_PATH}")
        return frame
    for face_landmarks in landmarks:
        nose_tip = face_landmarks[1]
        left_mouth_corner = face_landmarks[61]
        right_mouth_corner = face_landmarks[291]
        mouth_width = int(np.linalg.norm(np.array(right_mouth_corner) - np.array(left_mouth_corner)))
        mustache_width = int(mouth_width * 1.5)
        aspect_ratio = mustache.shape[0] / mustache.shape[1]
        mustache_height = int(mustache_width * aspect_ratio)
        resized_mustache = cv2.resize(mustache, (mustache_width, mustache_height), interpolation=cv2.INTER_AREA)
        mouth_delta_x = right_mouth_corner[0] - left_mouth_corner[0]
        mouth_delta_y = right_mouth_corner[1] - left_mouth_corner[1]
        angle = -np.degrees(np.arctan2(mouth_delta_y, mouth_delta_x))
        M = cv2.getRotationMatrix2D((mustache_width // 2, mustache_height // 2), angle, 1.0)
        rotated_mustache = cv2.warpAffine(resized_mustache, M, (mustache_width, mustache_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        center = np.mean([nose_tip, left_mouth_corner, right_mouth_corner], axis=0).astype(int)
        top_left = (int(center[0] - mustache_width / 2), int(nose_tip[1] - mustache_height * 0.2))
        top_left_y = max(0, top_left[1])
        bottom_right_y = min(frame.shape[0], top_left[1] + mustache_height)
        top_left_x = max(0, top_left[0])
        bottom_right_x = min(frame.shape[1], top_left[0] + mustache_width)
        roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        mustache_roi = rotated_mustache[top_left_y - top_left[1]:bottom_right_y - top_left[1], top_left_x - top_left[0]:bottom_right_x - top_left[0]]
        for i in range(mustache_roi.shape[0]):
            for j in range(mustache_roi.shape[1]):
                if mustache_roi[i, j, 3] > 0:
                    roi[i, j] = mustache_roi[i, j, :3]
    return frame

def apply_hairstyle_filter(frame, landmarks):
    if not landmarks:
        return frame
    
    hairstyle = cv2.imread(HAIRSTYLE_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if hairstyle is None:
        print(f"Error: Unable to load hairstyle image from {HAIRSTYLE_IMAGE_PATH}")
        return frame
    
    for face_landmarks in landmarks:
        if len(face_landmarks) < 468:  # Ensure we have enough landmarks
            continue
        
        forehead_left = face_landmarks[75]
        forehead_right = face_landmarks[294]
        forehead_top = face_landmarks[10]  # Top of the forehead
        
        forehead_width = int(np.linalg.norm(np.array(forehead_right) - np.array(forehead_left)))
        hairstyle_width = int(forehead_width * 1.5)
        aspect_ratio = hairstyle.shape[0] / hairstyle.shape[1]
        hairstyle_height = int(hairstyle_width * aspect_ratio)
        
        resized_hairstyle = cv2.resize(hairstyle, (hairstyle_width, hairstyle_height), interpolation=cv2.INTER_AREA)
        
        center_forehead_x = int((forehead_left[0] + forehead_right[0]) / 2)
        center_forehead_y = int(forehead_top[1] - hairstyle_height * 0.7)
        
        top_left_x = max(0, center_forehead_x - hairstyle_width // 2)
        top_left_y = max(0, center_forehead_y)
        
        bottom_right_y = min(frame.shape[0], top_left_y + hairstyle_height)
        bottom_right_x = min(frame.shape[1], top_left_x + hairstyle_width)
        
        roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        hairstyle_roi = resized_hairstyle[:bottom_right_y - top_left_y, :bottom_right_x - top_left_x]
        
        # Create a mask from the alpha channel
        mask = hairstyle_roi[:, :, 3] > 0
        
        # Apply the hairstyle only where the mask is True
        roi[mask] = hairstyle_roi[mask, :3]
    
    return frame

def apply_face_mask_filter(frame, landmarks):
    if not landmarks:
        return frame
    face_mask = cv2.imread(FACE_MASK_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if face_mask is None:
        print(f"Error: Unable to load face mask image from {FACE_MASK_IMAGE_PATH}")
        return frame
    for face_landmarks in landmarks:
        chin = face_landmarks[152]
        left_cheek = face_landmarks[234]
        right_cheek = face_landmarks[454]
        mask_width = int(np.linalg.norm(np.array(right_cheek) - np.array(left_cheek)) * 1.2)
        aspect_ratio = face_mask.shape[0] / face_mask.shape[1]
        mask_height = int(mask_width * aspect_ratio)
        resized_mask = cv2.resize(face_mask, (mask_width, mask_height), interpolation=cv2.INTER_AREA)
        mask_center_x = int((left_cheek[0] + right_cheek[0]) / 2)
        mask_center_y = int((chin[1] + (left_cheek[1] + right_cheek[1]) / 2) / 2)
        top_left_x = mask_center_x - mask_width // 2
        top_left_y = mask_center_y - mask_height // 2
        top_left_y = max(0, top_left_y)
        bottom_right_y = min(frame.shape[0], top_left_y + mask_height)
        top_left_x = max(0, top_left_x)
        bottom_right_x = min(frame.shape[1], top_left_x + mask_width)
        roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        mask_roi = resized_mask[:bottom_right_y - top_left_y, :bottom_right_x - top_left_x]
        for i in range(mask_roi.shape[0]):
            for j in range(mask_roi.shape[1]):
                if mask_roi[i, j, 3] > 0:
                    roi[i, j] = mask_roi[i, j, :3]
    return frame