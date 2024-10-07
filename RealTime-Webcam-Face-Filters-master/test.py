import cv2
import json
import numpy as np
from src.webcam_constants import WEBCAM_INDEX, WINDOW_NAME, FRAME_WAIT_KEY
from src.facial_landmark_detection import detect_facial_landmarks


def load_saved_landmarks(json_file="averaged_landmarks.json"):
    try:
        with open(json_file, "r") as file:
            data = json.load(file)

            # Check if the structure is a dictionary with "average_landmarks" key
            if isinstance(data, dict) and "average_landmarks" in data:
                return np.array(data["average_landmarks"])
            else:
                print("Error: JSON data does not contain 'average_landmarks' key.")
                return None
    except FileNotFoundError:
        print(f"Error: The file '{json_file}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file}'.")
        return None


def load_average_landmarks(json_file="averaged_landmarks.json"):
    try:
        with open(json_file, "r") as file:
            data = json.load(file)
            return np.array(data["average_landmarks"])
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: Could not load or decode average landmarks from '{json_file}'.")
        return None

def is_same_person(new_landmarks, avg_landmarks, threshold=0.15):
    if new_landmarks is None or avg_landmarks is None:
        return False

    # Convert landmarks to numpy arrays
    new_landmarks = np.array(new_landmarks)
    avg_landmarks = np.array(avg_landmarks)

    # Calculate mean Euclidean distance between corresponding landmarks
    distances = np.linalg.norm(new_landmarks - avg_landmarks, axis=1)
    mean_distance = np.mean(distances)

    print(f"Computed distance: {mean_distance}")
    return mean_distance < threshold

def check_if_same_person():
    saved_landmarks = load_saved_landmarks()  # Load saved landmarks from JSON
    if saved_landmarks is None:
        print("No saved person data to compare against.")
        return

    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
        return

    threshold = 500  # Set higher threshold for testing

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to read frame from webcam")
            break

        new_landmarks = detect_facial_landmarks(frame)  # Detect landmarks in the new frame

        if new_landmarks:
            # Check if the new person is the same as the saved person
            same_person = is_same_person(new_landmarks, saved_landmarks, threshold)
            status_text = "Same Person" if same_person else "Different Person"

            # Display the result on the frame
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if same_person else (0, 0, 255), 2)
        else:
            # No landmarks detected in the current frame
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(FRAME_WAIT_KEY) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Run the function to check if the current person is the same as the saved one
check_if_same_person()











# def load_average_landmarks(json_file="averaged_landmarks.json"):
#     try:
#         with open(json_file, "r") as file:
#             data = json.load(file)
#             return np.array(data["average_landmarks"])
#     except (FileNotFoundError, json.JSONDecodeError):
#         print(f"Error: Could not load or decode average landmarks from '{json_file}'.")
#         return None

# def is_same_person(new_landmarks, avg_landmarks, threshold=0.15):
#     if new_landmarks is None or avg_landmarks is None:
#         return False

#     # Convert landmarks to numpy arrays
#     new_landmarks = np.array(new_landmarks)
#     avg_landmarks = np.array(avg_landmarks)

#     # Calculate mean Euclidean distance between corresponding landmarks
#     distances = np.linalg.norm(new_landmarks - avg_landmarks, axis=1)
#     mean_distance = np.mean(distances)

#     print(f"Computed distance: {mean_distance}")
#     return mean_distance < threshold
