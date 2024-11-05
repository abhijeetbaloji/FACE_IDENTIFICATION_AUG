#v1 only face front

# import cv2
# import json
# import numpy as np
# from src.webcam_constants import (
#     WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY,
#     FILTER_NONE_KEY, FILTER_LANDMARK_KEY, FILTER_BLUR_KEY,
#     FILTER_SUNGLASSES_KEY, FILTER_MUSTACHE_KEY, FILTER_HAIRSTYLE_KEY,
#     FILTER_FACE_MASK_KEY, MENU_TEXT, MENU_POSITION, MENU_FONT,
#     MENU_FONT_SCALE, MENU_FONT_THICKNESS, MENU_COLOR
# )
# from src.facial_landmark_detection import detect_facial_landmarks, draw_facial_landmarks
# from src.face_filters import (
#     apply_blur_filter, apply_sunglasses_filter,
#     apply_mustache_filter, apply_hairstyle_filter,
#     apply_face_mask_filter
# )

# def open_webcam_with_filter_switching():
#     video_capture = cv2.VideoCapture(WEBCAM_INDEX)
#     if not video_capture.isOpened():
#         print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
#         return

#     current_filter = FILTER_NONE_KEY
#     frames_landmarks = []  # List to store each frame's landmarks

#     # Counter-based condition to auto-exit after a certain number of frames
#     frame_count = 0
#     max_frames = 300  # Define max frames to capture before exiting

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             print("Error: Unable to read frame from webcam")
#             break

#         landmarks = []
#         # Detect landmarks if any filter requiring landmarks is selected
#         if current_filter in [FILTER_LANDMARK_KEY, FILTER_BLUR_KEY, FILTER_SUNGLASSES_KEY, FILTER_MUSTACHE_KEY, FILTER_HAIRSTYLE_KEY, FILTER_FACE_MASK_KEY]:
#             landmarks = detect_facial_landmarks(frame)

#         # Store the landmarks for averaging later
#         if landmarks:
#             frames_landmarks.append(np.array(landmarks))

#         # Apply the selected filter to the frame
#         if current_filter == FILTER_LANDMARK_KEY:
#             frame = draw_facial_landmarks(frame, landmarks)
#         elif current_filter == FILTER_BLUR_KEY:
#             frame = apply_blur_filter(frame, landmarks)
#         elif current_filter == FILTER_SUNGLASSES_KEY:
#             frame = apply_sunglasses_filter(frame, landmarks)
#         elif current_filter == FILTER_MUSTACHE_KEY:
#             frame = apply_mustache_filter(frame, landmarks)
#         elif current_filter == FILTER_HAIRSTYLE_KEY:
#             frame = apply_hairstyle_filter(frame, landmarks)
#         elif current_filter == FILTER_FACE_MASK_KEY:
#             frame = apply_face_mask_filter(frame, landmarks)

#         # Display filter menu
#         for i, line in enumerate(MENU_TEXT.split("\n")):
#             cv2.putText(frame, line, (MENU_POSITION[0], MENU_POSITION[1] + i * 20),
#                         MENU_FONT, MENU_FONT_SCALE, MENU_COLOR, MENU_FONT_THICKNESS)

#         cv2.imshow(WINDOW_NAME, frame)

#         # Key controls for filter selection and exit
#         key = cv2.waitKey(FRAME_WAIT_KEY) & 0xFF
#         if key == ord(EXIT_KEY):  # Press the exit key to break out of the loop
#             break
#         elif key == ord(FILTER_NONE_KEY):
#             current_filter = FILTER_NONE_KEY
#         elif key == ord(FILTER_LANDMARK_KEY):
#             current_filter = FILTER_LANDMARK_KEY
#         elif key == ord(FILTER_BLUR_KEY):
#             current_filter = FILTER_BLUR_KEY
#         elif key == ord(FILTER_SUNGLASSES_KEY):
#             current_filter = FILTER_SUNGLASSES_KEY
#         elif key == ord(FILTER_MUSTACHE_KEY):
#             current_filter = FILTER_MUSTACHE_KEY
#         elif key == ord(FILTER_HAIRSTYLE_KEY):
#             current_filter = FILTER_HAIRSTYLE_KEY
#         elif key == ord(FILTER_FACE_MASK_KEY):
#             current_filter = FILTER_FACE_MASK_KEY

#         # Increment frame count and auto-exit after max_frames
#         frame_count += 1
#         if frame_count >= max_frames:
#             print("Reached maximum frame count. Exiting...")
#             break

#     # Release video capture and close OpenCV windows
#     video_capture.release()
#     cv2.destroyAllWindows()

#     # Calculate average landmarks if landmarks were collected
#     if frames_landmarks:
#         avg_landmarks = np.mean(frames_landmarks, axis=0).tolist()
        
#         # Save averaged landmarks data to JSON file
#         with open("averaged_landmarks.json", "w") as json_file:
#             json.dump({"average_landmarks": avg_landmarks}, json_file, indent=4)
#         print("Averaged landmarks have been saved to 'averaged_landmarks.json'.")

#         # Display similarity score between current session's average landmarks and previously saved average landmarks
#         score = calculate_similarity_score(avg_landmarks, "averaged_landmarks.json")
#         print(f"Similarity Score: {score}")
#     else:
#         print("No landmarks were detected and saved.")

# def calculate_similarity_score(new_avg_landmarks, json_file="averaged_landmarks.json"):
#     try:
#         # Load previously saved average landmarks
#         with open(json_file, "r") as file:
#             saved_data = json.load(file)
#             saved_avg_landmarks = np.array(saved_data["average_landmarks"])

#         # Convert new_avg_landmarks to numpy array
#         new_avg_landmarks = np.array(new_avg_landmarks)

#         # Calculate mean Euclidean distance between corresponding landmarks
#         distances = np.linalg.norm(new_avg_landmarks - saved_avg_landmarks, axis=1)
#         mean_distance = np.mean(distances)
#         return mean_distance

#     except (FileNotFoundError, json.JSONDecodeError):
#         print(f"Error: Could not load or decode landmarks from '{json_file}'.")
#         return None

# # Run the function to capture landmarks, save their average, and display similarity score
# open_webcam_with_filter_switching()



#v2 left right front



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

        # Store landmarks based on face orientation
        if landmarks and orientation:
            frames_landmarks[orientation].append(np.array(landmarks))

        # Apply filter based on current filter selection
        if current_filter == FILTER_LANDMARK_KEY:
            frame = draw_facial_landmarks(frame, landmarks)

        # Display the filter menu
        for i, line in enumerate(MENU_TEXT.split("\n")):
            cv2.putText(frame, line, (MENU_POSITION[0], MENU_POSITION[1] + i * 20),
                        MENU_FONT, MENU_FONT_SCALE, MENU_COLOR, MENU_FONT_THICKNESS)

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

    # Calculate average landmarks for each orientation
    averaged_landmarks = {}
    for orientation, landmarks_list in frames_landmarks.items():
        if landmarks_list:
            averaged_landmarks[orientation] = np.mean(landmarks_list, axis=0).tolist()
        else:
            averaged_landmarks[orientation] = []

    # Save averaged landmarks data to JSON
    with open("averaged_landmarks_by_orientation.json", "w") as json_file:
        json.dump(averaged_landmarks, json_file, indent=4)
    print("Averaged landmarks by orientation have been saved to 'averaged_landmarks_by_orientation.json'.")

    # Display similarity score for each orientation (optional)
    for orientation in ["front", "left", "right"]:
        score = calculate_similarity_score(averaged_landmarks.get(orientation, []), orientation)
        print(f"Similarity Score for {orientation} orientation: {score}")

def calculate_similarity_score(new_avg_landmarks, orientation, json_file="averaged_landmarks_by_orientation.json"):
    try:
        with open(json_file, "r") as file:
            saved_data = json.load(file)
            saved_avg_landmarks = np.array(saved_data.get(orientation, []))

        if not saved_avg_landmarks.any():
            print(f"No saved landmarks found for {orientation} orientation.")
            return None

        new_avg_landmarks = np.array(new_avg_landmarks)
        distances = np.linalg.norm(new_avg_landmarks - saved_avg_landmarks, axis=1)
        mean_distance = np.mean(distances)
        return mean_distance

    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: Could not load or decode landmarks from '{json_file}'.")
        return None

# Run the function to capture multi-angle landmarks and save their averages
open_webcam_with_orientation_landmarks()
