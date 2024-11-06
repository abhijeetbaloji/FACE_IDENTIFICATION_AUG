# # # import cv2
# # # import json
# # # import numpy as np
# # # from src.webcam_constants import (
# # #     WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY,
# # #     FILTER_NONE_KEY, FILTER_LANDMARK_KEY, FILTER_BLUR_KEY,
# # #     FILTER_SUNGLASSES_KEY, FILTER_MUSTACHE_KEY, FILTER_HAIRSTYLE_KEY,
# # #     FILTER_FACE_MASK_KEY, MENU_TEXT, MENU_POSITION, MENU_FONT,
# # #     MENU_FONT_SCALE, MENU_FONT_THICKNESS, MENU_COLOR
# # # )
# # # from src.facial_landmark_detection import detect_facial_landmarks, draw_facial_landmarks, detect_face_orientation, get_yaw_angle_from_landmarks
# # # from src.face_filters import (
# # #     apply_blur_filter, apply_sunglasses_filter,
# # #     apply_mustache_filter, apply_hairstyle_filter,
# # #     apply_face_mask_filter
# # # )

# # # # Threshold for determining "Same Person" based on similarity score
# # # SIMILARITY_THRESHOLD = 1000

# # # # Function to open the webcam and process face orientation and landmarks
# # # def open_webcam_with_orientation_landmarks():
# # #     video_capture = cv2.VideoCapture(WEBCAM_INDEX)
# # #     if not video_capture.isOpened():
# # #         print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
# # #         return

# # #     current_filter = FILTER_NONE_KEY
# # #     frames_landmarks = {"front": [], "left": [], "right": []}
# # #     frame_count = 0
# # #     max_frames = 1000

# # #     while True:
# # #         ret, frame = video_capture.read()
# # #         if not ret:
# # #             print("Error: Unable to read frame from webcam")
# # #             break

# # #         # Detect landmarks and face orientation
# # #         landmarks = detect_facial_landmarks(frame)
# # #         #print(landmarks)
# # #         orientation = detect_face_orientation(landmarks)  # Returns "front", "left", or "right"
# # #         #print(orientation)
# # #         # Store landmarks based on face orientation
# # #         if landmarks and orientation:
# # #             frames_landmarks[orientation].append(np.array(landmarks))
# # #         #print(frames_landmarks)

# # #         # Apply the current filter
# # #         if current_filter == FILTER_LANDMARK_KEY:
# # #             frame = draw_facial_landmarks(frame, landmarks)
# # #         #print(frame)
# # #         #print(landmarks)

# # #         # Calculate the yaw angle if landmarks are detected
# # #         yaw_angle = None
# # #         if landmarks:
# # #             yaw_angle = get_yaw_angle_from_landmarks(landmarks[0])  # Use the first detected face
# # #             print(yaw_angle)
        
# # #         # Display the yaw angle on the frame
# # #         if yaw_angle is not None:
# # #             cv2.putText(frame, f"Yaw Angle: {yaw_angle:.2f}°", (10, 30),
# # #                         MENU_FONT, MENU_FONT_SCALE, MENU_COLOR, MENU_FONT_THICKNESS)

# # #         # Display the filter menu
# # #         for i, line in enumerate(MENU_TEXT.split("\n")):
# # #             cv2.putText(frame, line, (MENU_POSITION[0], MENU_POSITION[1] + i * 20),
# # #                         MENU_FONT, MENU_FONT_SCALE, MENU_COLOR, MENU_FONT_THICKNESS)

# # #         # Check similarity score and display "Same Person" or "Different Person" on frame
# # #         for i, orientation in enumerate(["front", "left", "right"]):
# # #             score = calculate_similarity_score(frames_landmarks.get(orientation, []), orientation)
# # #             if score is not None:
# # #                 identity_text = "Same Person" if score < SIMILARITY_THRESHOLD else "Different Person"
# # #                 score_text = f"{identity_text} ({orientation}): {score:.2f}"
# # #             else:
# # #                 score_text = f"No data for {orientation}"

# # #             cv2.putText(frame, score_text, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# # #         cv2.imshow(WINDOW_NAME, frame)

# # #         # Key controls for filter selection and exit
# # #         key = cv2.waitKey(FRAME_WAIT_KEY) & 0xFF
# # #         if key == ord(EXIT_KEY):
# # #             break
# # #         elif key == ord(FILTER_NONE_KEY):
# # #             current_filter = FILTER_NONE_KEY
# # #         elif key == ord(FILTER_LANDMARK_KEY):
# # #             current_filter = FILTER_LANDMARK_KEY
# # #         elif key == ord(FILTER_BLUR_KEY):
# # #             current_filter = FILTER_BLUR_KEY
# # #         elif key == ord(FILTER_SUNGLASSES_KEY):
# # #             current_filter = FILTER_SUNGLASSES_KEY
# # #         elif key == ord(FILTER_MUSTACHE_KEY):
# # #             current_filter = FILTER_MUSTACHE_KEY
# # #         elif key == ord(FILTER_HAIRSTYLE_KEY):
# # #             current_filter = FILTER_HAIRSTYLE_KEY
# # #         elif key == ord(FILTER_FACE_MASK_KEY):
# # #             current_filter = FILTER_FACE_MASK_KEY

# # #         # Increment frame count and auto-exit after max_frames
# # #         frame_count += 1
# # #         if frame_count >= max_frames:
# # #             print("Reached maximum frame count. Exiting...")
# # #             break

# # #     # Release video capture and close OpenCV windows
# # #     video_capture.release()
# # #     cv2.destroyAllWindows()

# # # # Function to calculate similarity score
# # # def calculate_similarity_score(new_landmarks_list, orientation, json_file="averaged_landmarks_by_orientation.json"):
# # #     try:
# # #         with open(json_file, "r") as file:
# # #             saved_data = json.load(file)
# # #             saved_avg_landmarks = np.array(saved_data.get(orientation, []))

# # #         # Ensure we have saved landmarks and new landmarks for comparison
# # #         if not saved_avg_landmarks.any() or not new_landmarks_list:
# # #             print(f"No landmarks data found for {orientation} orientation.")
# # #             return None

# # #         # Calculate mean of new landmarks
# # #         new_avg_landmarks = np.mean(np.array(new_landmarks_list), axis=0)

# # #         # Calculate mean Euclidean distance between corresponding landmarks
# # #         distances = np.linalg.norm(new_avg_landmarks - saved_avg_landmarks, axis=1)
# # #         mean_distance = np.mean(distances)
# # #         return mean_distance

# # #     except (FileNotFoundError, json.JSONDecodeError):
# # #         print(f"Error: Could not load or decode landmarks from '{json_file}'.")
# # #         return None

# # # # Run the function to test landmarks and similarity index
# # # open_webcam_with_orientation_landmarks()


# # import cv2
# # import json
# # import numpy as np
# # import mediapipe as mp
# # from src.webcam_constants import (
# #     WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY,
# #     FILTER_NONE_KEY, FILTER_LANDMARK_KEY, FILTER_BLUR_KEY,
# #     FILTER_SUNGLASSES_KEY, FILTER_MUSTACHE_KEY, FILTER_HAIRSTYLE_KEY,
# #     FILTER_FACE_MASK_KEY, MENU_TEXT, MENU_POSITION, MENU_FONT,
# #     MENU_FONT_SCALE, MENU_FONT_THICKNESS, MENU_COLOR
# # )

# # # Initialize Mediapipe FaceMesh
# # mp_face_mesh = mp.solutions.face_mesh
# # mp_drawing = mp.solutions.drawing_utils

# # # Threshold for determining "Same Person" based on similarity score
# # SIMILARITY_THRESHOLD = 6500

# # # Detect facial landmarks using Mediapipe
# # def detect_facial_landmarks(frame):
# #     with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
# #         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         results = face_mesh.process(image)

# #         if results.multi_face_landmarks:
# #             landmarks = []
# #             for landmark in results.multi_face_landmarks[0].landmark:
# #                 landmarks.append((landmark.x, landmark.y))
# #             return np.array(landmarks)  # Return as NumPy array
# #         else:
# #             return None



# # # Placeholder function to detect face orientation based on landmarks
# # def detect_face_orientation(landmarks):
# #     if landmarks is None or len(landmarks) == 0:
# #         return None

# #     # Define landmark indices for eyes and nose
# #     LEFT_EYE = 33  # You can adjust these indices based on your needs
# #     RIGHT_EYE = 263
# #     NOSE = 1

# #     left_eye_x = landmarks[LEFT_EYE][0]
# #     right_eye_x = landmarks[RIGHT_EYE][0]
# #     nose_x = landmarks[NOSE][0]

# #     # Determine face orientation based on eye positions
# #     if nose_x < (left_eye_x + right_eye_x) / 2:
# #         return "right"  # Face turned right
# #     elif nose_x > (left_eye_x + right_eye_x) / 2:
# #         return "left"  # Face turned left
# #     else:
# #         return "front"  # Face is facing forward





# # # Function to draw facial landmarks on the frame
# # def draw_facial_landmarks(frame, landmarks):
# #     if landmarks is not None:
# #         for landmark in landmarks:
# #             x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
# #             cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
# #     return frame

# # # Function to calculate similarity score, allowing for partial landmark matches
# # def calculate_similarity_score(detected_landmarks, stored_landmarks, orientation, json_file="averaged_landmarks_by_orientation.json"):
# #     try:
# #         with open(json_file, "r") as file:
# #             saved_data = json.load(file)
# #             saved_avg_landmarks = np.array(saved_data.get(orientation, []))

# #         # Ensure we have saved landmarks and new landmarks for comparison
# #         if saved_avg_landmarks.size == 0 or detected_landmarks is None:
# #             print(f"No landmarks data found for {orientation} orientation.")
# #             return None

# #         # Identify valid indices in saved landmarks
# #         valid_indices = [i for i in range(len(saved_avg_landmarks)) if not np.isnan(saved_avg_landmarks[i]).any()]
        
# #         # If no valid indices found, return None
# #         if not valid_indices:
# #             print(f"No valid landmarks found for comparison in {orientation}.")
# #             return None

# #         # Filter valid new landmarks based on detected landmarks
# #         valid_new_landmarks = detected_landmarks[valid_indices]
# #         valid_saved_landmarks = saved_avg_landmarks[valid_indices]

# #         # Calculate mean Euclidean distance between corresponding landmarks
# #         distances = np.linalg.norm(valid_new_landmarks - valid_saved_landmarks, axis=1)
# #         mean_distance = np.mean(distances)

# #         return mean_distance

# #     except (FileNotFoundError, json.JSONDecodeError):
# #         print(f"Error: Could not load or decode landmarks from '{json_file}'.")
# #         return None

# # # Open the webcam and process face orientation and landmarks
# # def open_webcam_with_orientation_landmarks():
# #     video_capture = cv2.VideoCapture(WEBCAM_INDEX)
# #     if not video_capture.isOpened():
# #         print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
# #         return

# #     current_filter = FILTER_NONE_KEY
# #     frames_landmarks = {"front": [], "left": [], "right": []}
# #     frame_count = 0
# #     max_frames = 1000

# #     while True:
# #         ret, frame = video_capture.read()
# #         if not ret:
# #             print("Error: Unable to read frame from webcam")
# #             break

# #         # Detect landmarks and face orientation
# #         landmarks = detect_facial_landmarks(frame)
# #         print("Detected landmarks:", landmarks)  # Debugging output
# #         orientation = detect_face_orientation(landmarks)  # Returns "front", "left", or "right"

# #         # Store landmarks based on face orientation
# #         if landmarks is not None and orientation:
# #             frames_landmarks[orientation].append(np.array(landmarks))

# #         # Apply the current filter
# #         if current_filter == FILTER_LANDMARK_KEY:
# #             frame = draw_facial_landmarks(frame, landmarks)

# #         # Check similarity score and display "Same Person" or "Different Person" on frame
# #         for i, orientation in enumerate(["front", "left", "right"]):
# #             # Use front landmarks if the current orientation does not have landmarks
# #             if orientation == "left" or orientation == "right":
# #                 comparison_landmarks = frames_landmarks.get("front", [])
# #             else:
# #                 comparison_landmarks = frames_landmarks.get(orientation, [])
            
# #             score = calculate_similarity_score(landmarks, comparison_landmarks, orientation)
# #             if score is not None:
# #                 identity_text = "Same Person" if score < SIMILARITY_THRESHOLD else "Different Person"
# #                 score_text = f"{identity_text} ({orientation}): {score:.2f}"
# #             else:
# #                 score_text = f"No data for {orientation}"

# #             cv2.putText(frame, score_text, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# #         cv2.imshow(WINDOW_NAME, frame)

# #         # Key controls for filter selection and exit
# #         key = cv2.waitKey(FRAME_WAIT_KEY) & 0xFF
# #         if key == ord(EXIT_KEY):
# #             break
# #         elif key == ord(FILTER_NONE_KEY):
# #             current_filter = FILTER_NONE_KEY
# #         elif key == ord(FILTER_LANDMARK_KEY):
# #             current_filter = FILTER_LANDMARK_KEY
# #         elif key == ord(FILTER_BLUR_KEY):
# #             current_filter = FILTER_BLUR_KEY
# #         elif key == ord(FILTER_SUNGLASSES_KEY):
# #             current_filter = FILTER_SUNGLASSES_KEY
# #         elif key == ord(FILTER_MUSTACHE_KEY):
# #             current_filter = FILTER_MUSTACHE_KEY
# #         elif key == ord(FILTER_HAIRSTYLE_KEY):
# #             current_filter = FILTER_HAIRSTYLE_KEY
# #         elif key == ord(FILTER_FACE_MASK_KEY):
# #             current_filter = FILTER_FACE_MASK_KEY

# #         # Increment frame count and auto-exit after max_frames
# #         frame_count += 1
# #         if frame_count >= max_frames:
# #             print("Reached maximum frame count. Exiting...")
# #             break

# #     # Release video capture and close OpenCV windows
# #     video_capture.release()
# #     cv2.destroyAllWindows()

# # # Run the function to test landmarks and similarity index
# # open_webcam_with_orientation_landmarks()



# import cv2
# import json
# import numpy as np
# import mediapipe as mp
# from src.webcam_constants import (
#     WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY,
#     FILTER_NONE_KEY, FILTER_LANDMARK_KEY, FILTER_BLUR_KEY,
#     FILTER_SUNGLASSES_KEY, FILTER_MUSTACHE_KEY, FILTER_HAIRSTYLE_KEY,
#     FILTER_FACE_MASK_KEY, MENU_TEXT, MENU_POSITION, MENU_FONT,
#     MENU_FONT_SCALE, MENU_FONT_THICKNESS, MENU_COLOR
# )

# # Initialize Mediapipe FaceMesh
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# # Threshold for determining "Same Person" based on similarity score
# SIMILARITY_THRESHOLD = 0.1  # Adjust this threshold based on testing

# # Detect facial landmarks using Mediapipe
# def detect_facial_landmarks(frame):
#     with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(image)

#         if results.multi_face_landmarks:
#             landmarks = []
#             for landmark in results.multi_face_landmarks[0].landmark:
#                 landmarks.append((landmark.x, landmark.y))
#             return np.array(landmarks)  # Return as NumPy array
#         else:
#             return None

# # Function to detect face orientation based on landmarks
# def detect_face_orientation(landmarks):
#     if landmarks is None or len(landmarks) == 0:
#         return None

#     # Define landmark indices for eyes and nose
#     LEFT_EYE = 33  # You can adjust these indices based on your needs
#     RIGHT_EYE = 263
#     NOSE = 1

#     left_eye_x = landmarks[LEFT_EYE][0]
#     right_eye_x = landmarks[RIGHT_EYE][0]
#     nose_x = landmarks[NOSE][0]

#     # Determine face orientation based on eye positions
#     if nose_x < (left_eye_x + right_eye_x) / 2:
#         return "right"  # Face turned right
#     elif nose_x > (left_eye_x + right_eye_x) / 2:
#         return "left"  # Face turned left
#     else:
#         return "front"  # Face is facing forward

# # Function to draw facial landmarks on the frame
# def draw_facial_landmarks(frame, landmarks):
#     if landmarks is not None:
#         for landmark in landmarks:
#             x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
#             cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#     return frame

# # Function to calculate similarity score using Mean Squared Error
# def calculate_similarity_score(detected_landmarks, stored_landmarks, orientation, json_file="averaged_landmarks_by_orientation.json"):
#     try:
#         with open(json_file, "r") as file:
#             saved_data = json.load(file)
#             saved_avg_landmarks = np.array(saved_data.get(orientation, []))

#         if saved_avg_landmarks.size == 0 or detected_landmarks is None:
#             print(f"No landmarks data found for {orientation} orientation.")
#             return None

#         # Filter out NaN values from detected landmarks
#         detected_landmarks = detected_landmarks[~np.isnan(detected_landmarks).any(axis=1)]
#         valid_indices = [i for i in range(len(saved_avg_landmarks)) if not np.isnan(saved_avg_landmarks[i]).any()]

#         if not valid_indices:
#             print(f"No valid landmarks found for comparison in {orientation}.")
#             return None

#         valid_saved_landmarks = saved_avg_landmarks[valid_indices]

#         # Calculate Mean Squared Error between detected and saved landmarks
#         distances = np.mean((detected_landmarks - valid_saved_landmarks) ** 2, axis=1)
#         mean_distance = np.mean(distances)

#         return mean_distance

#     except (FileNotFoundError, json.JSONDecodeError):
#         print(f"Error: Could not load or decode landmarks from '{json_file}'.")
#         return None

# # Open the webcam and process face orientation and landmarks
# def open_webcam_with_orientation_landmarks():
#     video_capture = cv2.VideoCapture(WEBCAM_INDEX)
#     if not video_capture.isOpened():
#         print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
#         return

#     current_filter = FILTER_NONE_KEY
#     frames_landmarks = {"front": [], "left": [], "right": []}
#     frame_count = 0
#     max_frames = 1000

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             print("Error: Unable to read frame from webcam")
#             break

#         # Detect landmarks and face orientation
#         landmarks = detect_facial_landmarks(frame)
#         print("Detected landmarks:", landmarks)  # Debugging output
#         orientation = detect_face_orientation(landmarks)  # Returns "front", "left", or "right"

#         # Store landmarks based on face orientation
#         if landmarks is not None and orientation:
#             frames_landmarks[orientation].append(np.array(landmarks))

#         # Apply the current filter
#         if current_filter == FILTER_LANDMARK_KEY:
#             frame = draw_facial_landmarks(frame, landmarks)

#         # Check similarity score and display "Same Person" or "Different Person" on frame
#         for i, orientation in enumerate(["front", "left", "right"]):
#             # Use front landmarks if the current orientation does not have landmarks
#             if orientation == "left" or orientation == "right":
#                 comparison_landmarks = frames_landmarks.get("front", [])
#             else:
#                 comparison_landmarks = frames_landmarks.get(orientation, [])

#             # Calculate similarity score
#             if comparison_landmarks:
#                 last_landmarks = comparison_landmarks[-1]  # Use the last stored landmarks
#                 score = calculate_similarity_score(landmarks, last_landmarks, orientation)
#                 if score is not None:
#                     identity_text = "Same Person" if score < SIMILARITY_THRESHOLD else "Different Person"
#                     score_text = f"{identity_text} ({orientation}): {score:.4f}"
#                 else:
#                     score_text = f"No data for {orientation}"
#             else:
#                 score_text = f"No data for {orientation}"

#             cv2.putText(frame, score_text, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#         cv2.imshow(WINDOW_NAME, frame)

#         # Key controls for filter selection and exit
#         key = cv2.waitKey(FRAME_WAIT_KEY) & 0xFF
#         if key == ord(EXIT_KEY):
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

# # Run the function to test landmarks and similarity index
# open_webcam_with_orientation_landmarks()

import cv2
import json
import numpy as np
import mediapipe as mp
from src.webcam_constants import (
    WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY,
    FILTER_NONE_KEY, FILTER_LANDMARK_KEY, FILTER_BLUR_KEY,
    FILTER_SUNGLASSES_KEY, FILTER_MUSTACHE_KEY, FILTER_HAIRSTYLE_KEY,
    FILTER_FACE_MASK_KEY, MENU_TEXT, MENU_POSITION, MENU_FONT,
    MENU_FONT_SCALE, MENU_FONT_THICKNESS, MENU_COLOR
)

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Threshold for determining "Same Person" based on similarity score
SIMILARITY_THRESHOLD = 0.1  # Adjust this threshold based on testing

# Detect facial landmarks using Mediapipe
def detect_facial_landmarks(frame):
    with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            landmarks = []
            for landmark in results.multi_face_landmarks[0].landmark:
                landmarks.append((landmark.x, landmark.y))
            return np.array(landmarks)  # Return as NumPy array
        else:
            return None

# Function to detect face orientation based on landmarks
def detect_face_orientation(landmarks):
    if landmarks is None or len(landmarks) == 0:
        return None

    # Define landmark indices for eyes and nose
    LEFT_EYE = 33  # You can adjust these indices based on your needs
    RIGHT_EYE = 263
    NOSE = 1

    left_eye_x = landmarks[LEFT_EYE][0]
    right_eye_x = landmarks[RIGHT_EYE][0]
    nose_x = landmarks[NOSE][0]

    # Determine face orientation based on eye positions
    if nose_x < (left_eye_x + right_eye_x) / 2:
        return "right"  # Face turned right
    elif nose_x > (left_eye_x + right_eye_x) / 2:
        return "left"  # Face turned left
    else:
        return "front"  # Face is facing forward

# Function to draw facial landmarks on the frame
def draw_facial_landmarks(frame, landmarks):
    if landmarks is not None:
        for landmark in landmarks:
            x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame

# Function to calculate similarity score using Mean Squared Error
def calculate_similarity_score(detected_landmarks, stored_landmarks, orientation, json_file="averaged_landmarks_by_orientation.json"):
    try:
        with open(json_file, "r") as file:
            saved_data = json.load(file)
            saved_avg_landmarks = np.array(saved_data.get(orientation, []))

        if saved_avg_landmarks.size == 0 or detected_landmarks is None:
            return None

        # Filter out NaN values from detected landmarks
        detected_landmarks = detected_landmarks[~np.isnan(detected_landmarks).any(axis=1)]
        valid_indices = [i for i in range(len(saved_avg_landmarks)) if not np.isnan(saved_avg_landmarks[i]).any()]

        if not valid_indices:
            return None

        valid_saved_landmarks = saved_avg_landmarks[valid_indices]

        # Calculate Mean Squared Error between detected and saved landmarks
        distances = np.mean((detected_landmarks - valid_saved_landmarks) ** 2, axis=1)
        mean_distance = np.mean(distances)

        return mean_distance

    except (FileNotFoundError, json.JSONDecodeError):
        return None

# Open the webcam and process face orientation and landmarks
def open_webcam_with_orientation_landmarks():
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
        return

    current_filter = FILTER_NONE_KEY
    frames_landmarks = {"front": [], "left": [], "right": []}
    frame_count = 0
    max_frames = 1000

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to read frame from webcam")
            break

        # Detect landmarks and face orientation
        landmarks = detect_facial_landmarks(frame)
        orientation = detect_face_orientation(landmarks)  # Returns "front", "left", or "right"

        # Store landmarks based on face orientation
        if landmarks is not None and orientation:
            frames_landmarks[orientation].append(np.array(landmarks))

        # Apply the current filter
        if current_filter == FILTER_LANDMARK_KEY:
            frame = draw_facial_landmarks(frame, landmarks)

        # Check similarity score and display "Same Person" or "Different Person" on frame
        for i, orientation in enumerate(["front", "left", "right"]):
            if orientation == "left" or orientation == "right":
                comparison_landmarks = frames_landmarks.get("front", [])
            else:
                comparison_landmarks = frames_landmarks.get(orientation, [])

            # Calculate similarity score
            if comparison_landmarks:
                last_landmarks = comparison_landmarks[-1]  # Use the last stored landmarks
                score = calculate_similarity_score(landmarks, last_landmarks, orientation)
                if score is not None:
                    if score < SIMILARITY_THRESHOLD:
                        print("Same Person")  # Print only if the score indicates the same person
                    # Display the score on the frame (optional)
                    score_text = f"Score: {score:.4f}"
                    cv2.putText(frame, score_text, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    print("diff")
                    print(score)

        # Show the frame with the drawn text
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

# Run the function to test landmarks and similarity index
open_webcam_with_orientation_landmarks()
