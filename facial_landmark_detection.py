import cv2
import mediapipe as mp
import numpy as np
from src.webcam_constants import FACIAL_LANDMARK_WINDOW_NAME

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def detect_facial_landmarks(frame):
    """
    Detect facial landmarks in the given frame using MediaPipe Face Mesh.

    Args:
        frame (numpy.ndarray): The frame from the webcam capture.

    Returns:
        landmarks (list): A list of facial landmarks.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks.append(
                [
                    (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                    for landmark in face_landmarks.landmark
                ]
            )
    return landmarks

def draw_facial_landmarks(frame, landmarks):
    """
    Draw facial landmarks on the frame.

    Args:
        frame (numpy.ndarray): The frame from the webcam capture.
        landmarks (list): A list of facial landmarks.

    Returns:
        frame (numpy.ndarray): The frame with landmarks drawn.
    """
    for face_landmarks in landmarks:
        for x, y in face_landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    return frame

def detect_face_orientation(landmarks):
    """
    Detect the orientation of the face (front, left, or right) based on landmarks.

    Args:
        landmarks (list): A list of facial landmarks.

    Returns:
        str: Orientation of the face ('front', 'left', 'right').
    """
    if not landmarks or len(landmarks[0]) == 0:
        return None  # No landmarks detected

    face_landmarks = landmarks[0]  # Assume a single face for simplicity

    # Get coordinates for left eye, right eye, and nose tip
    nose_tip = face_landmarks[1]      # Nose tip
    left_eye = face_landmarks[33]     # Left eye outer corner
    right_eye = face_landmarks[263]   # Right eye outer corner

    # Calculate the horizontal distance between nose and eyes
    left_dist = nose_tip[0] - left_eye[0]
    right_dist = right_eye[0] - nose_tip[0]

    # Determine orientation based on distances
    if abs(left_dist - right_dist) < 10:  # Adjust threshold as needed
        return "front"
    elif left_dist > right_dist:
        return "left"
    else:
        return "right"

def get_yaw_angle_from_landmarks(landmarks):
    """
    Calculate the yaw angle based on the detected facial landmarks.

    Args:
        landmarks (list): A list of (x, y) coordinates for facial landmarks.

    Returns:
        float: The calculated yaw angle in degrees.
    """
    # Ensure landmarks contain sufficient points, e.g., 468 for a typical facial landmark model
    if len(landmarks) < 468:
        return None  # Or handle the error as appropriate

    # Example landmark indices for the nose tip and chin
    nose_tip = landmarks[1]  # Nose tip landmark (index may vary)
    chin = landmarks[8]      # Chin landmark (index may vary)

    # Calculate the yaw angle using the x-coordinates of the nose tip and chin
    yaw_angle = np.arctan2(chin[0] - nose_tip[0], chin[1] - nose_tip[1]) * (180.0 / np.pi)

    return yaw_angle

