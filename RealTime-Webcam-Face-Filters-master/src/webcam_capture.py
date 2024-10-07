import cv2
from src.webcam_constants import (
    WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY,
    FILTER_NONE_KEY, FILTER_LANDMARK_KEY, FILTER_BLUR_KEY,
    FILTER_SUNGLASSES_KEY, FILTER_MUSTACHE_KEY, FILTER_HAIRSTYLE_KEY,
    FILTER_FACE_MASK_KEY, MENU_TEXT, MENU_POSITION, MENU_FONT,
    MENU_FONT_SCALE, MENU_FONT_THICKNESS, MENU_COLOR
)
from src.facial_landmark_detection import detect_facial_landmarks, draw_facial_landmarks
from src.face_filters import (
    apply_blur_filter, apply_sunglasses_filter,
    apply_mustache_filter, apply_hairstyle_filter,
    apply_face_mask_filter
)

def open_webcam_with_filter_switching():
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
        return

    current_filter = FILTER_NONE_KEY
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to read frame from webcam")
            break

        landmarks = []
        if current_filter in [FILTER_LANDMARK_KEY, FILTER_BLUR_KEY, FILTER_SUNGLASSES_KEY, FILTER_MUSTACHE_KEY, FILTER_HAIRSTYLE_KEY, FILTER_FACE_MASK_KEY]:
            landmarks = detect_facial_landmarks(frame)

        if current_filter == FILTER_LANDMARK_KEY:
            frame = draw_facial_landmarks(frame, landmarks)
        elif current_filter == FILTER_BLUR_KEY:
            frame = apply_blur_filter(frame, landmarks)
        elif current_filter == FILTER_SUNGLASSES_KEY:
            frame = apply_sunglasses_filter(frame, landmarks)
        elif current_filter == FILTER_MUSTACHE_KEY:
            frame = apply_mustache_filter(frame, landmarks)
        elif current_filter == FILTER_HAIRSTYLE_KEY:
            frame = apply_hairstyle_filter(frame, landmarks)
        elif current_filter == FILTER_FACE_MASK_KEY:
            frame = apply_face_mask_filter(frame, landmarks)

        for i, line in enumerate(MENU_TEXT.split("\n")):
            cv2.putText(frame, line, (MENU_POSITION[0], MENU_POSITION[1] + i * 20),
                        MENU_FONT, MENU_FONT_SCALE, MENU_COLOR, MENU_FONT_THICKNESS)

        cv2.imshow(WINDOW_NAME, frame)

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

    video_capture.release()
    cv2.destroyAllWindows()