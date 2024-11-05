import cv2
import json
import numpy as np
from src.webcam_constants import (
    WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY,
    FILTER_NONE_KEY, FILTER_LANDMARK_KEY, FILTER_BLUR_KEY,
    FILTER_SUNGLASSES_KEY, FILTER_MUSTACHE_KEY, FILTER_HAIRSTYLE_KEY,
    FILTER_FACE_MASK_KEY, MENU_TEXT, MENU_POSITION, MENU_FONT,
    MENU_FONT_SCALE, MENU_FONT_THICKNESS, MENU_COLOR
)
from src.facial_landmark_detection import detect_facial_landmarks, draw_facial_landmarks, detect_face_orientation
from src.face_filters import (
    apply_blur_filter, apply_sunglasses_filter,
    apply_mustache_filter, apply_hairstyle_filter,
    apply_face_mask_filter
)

# Threshold for determining "Same Person" based on similarity score
SIMILARITY_THRESHOLD = 500

def open_webcam_with_orientation_landmarks():
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
        return

    current_filter = FILTER_NONE_KEY
    frames_landmarks = {"front": [], "left": [], "right": []}
    frame_count = 0
    max_frames = 300

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to read frame from webcam")
            break

        # Detect landmarks and face orientation
        landmarks = detect_facial_landmarks(frame)
        orientation = detect_face_orientation(landmarks)  # Returns "front", "left", or "right"

        # Store landmarks based on face orientation for testing only
        if landmarks and orientation:
            frames_landmarks[orientation].append(np.array(landmarks))

        # Apply filter based on current filter selection
        if current_filter == FILTER_LANDMARK_KEY:
            frame = draw_facial_landmarks(frame, landmarks)

        # Display the filter menu
        for i, line in enumerate(MENU_TEXT.split("\n")):
            cv2.putText(frame, line, (MENU_POSITION[0], MENU_POSITION[1] + i * 20),
                        MENU_FONT, MENU_FONT_SCALE, MENU_COLOR, MENU_FONT_THICKNESS)

        # Check similarity score and display "Same Person" or "Different Person" on frame
        for i, orientation in enumerate(["front", "left", "right"]):
            score = calculate_similarity_score(frames_landmarks.get(orientation, []), orientation)
            if score is not None:
                identity_text = "Same Person" if score < SIMILARITY_THRESHOLD else "Different Person"
                score_text = f"{identity_text} ({orientation}): {score:.2f}"
            else:
                score_text = f"No data for {orientation}"
            
            cv2.putText(frame, score_text, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        # Key controls for filter selection and exit
        key = cv2.waitKey(FRAME_WAIT_KEY) & 0xFF
        if key == ord(EXIT_KEY):
            break
        elif key == ord(FILTER_NONE_KEY):
            current_filter = FILTER_NONE_KEY
        elif key == ord(FILTER_LANDMARK_KEY):
            current_filter = FILTER_LANDMARK_KEY
        elif key == ord(FILTER_BLUR_KEY):
            current_filter = FILTER_BLUR_KEY
        elif key == ord(FILTER_SUNGLASSES_KEY):
            current_filter = FILTER_SUNGLASSES_KEY
        elif key == ord(FILTER_MUSTACHE_KEY):
            current_filter = FILTER_MUSTACHE_KEY
        elif key == ord(FILTER_HAIRSTYLE_KEY):
            current_filter = FILTER_HAIRSTYLE_KEY
        elif key == ord(FILTER_FACE_MASK_KEY):
            current_filter = FILTER_FACE_MASK_KEY

        # Increment frame count and auto-exit after max_frames
        frame_count += 1
        if frame_count >= max_frames:
            print("Reached maximum frame count. Exiting...")
            break

    # Release video capture and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

def calculate_similarity_score(new_landmarks_list, orientation, json_file="averaged_landmarks_by_orientation.json"):
    try:
        with open(json_file, "r") as file:
            saved_data = json.load(file)
            saved_avg_landmarks = np.array(saved_data.get(orientation, []))

        # Ensure we have saved landmarks and new landmarks for comparison
        if not saved_avg_landmarks.any() or not new_landmarks_list:
            print(f"No landmarks data found for {orientation} orientation.")
            return None

        # Calculate mean of new landmarks
        new_avg_landmarks = np.mean(np.array(new_landmarks_list), axis=0)
        
        # Calculate mean Euclidean distance between corresponding landmarks
        distances = np.linalg.norm(new_avg_landmarks - saved_avg_landmarks, axis=1)
        mean_distance = np.mean(distances)
        return mean_distance

    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: Could not load or decode landmarks from '{json_file}'.")
        return None

# Run the function to only test against saved landmarks
open_webcam_with_orientation_landmarks()
