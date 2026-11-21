
import cv2
import mediapipe as mp
import time
import math
import os
import pyautogui
from collections import deque

# -------------------------------
# SETTINGS
# -------------------------------

MODEL_PATH = os.path.join("models", "hand_landmarker.task")

COOLDOWN = 0.4
pyautogui.PAUSE = 0.02

# -------------------------------
# CHECK MODEL
# -------------------------------

if not os.path.exists(MODEL_PATH):
    print("ERROR: Hand model not found!")
    raise SystemExit

# -------------------------------
# MEDIAPIPE INITIALIZATION
# -------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# -------------------------------
# CAMERA
# -------------------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    detector.close()
    raise SystemExit

# -------------------------------
# FINGER DETECTION
# -------------------------------

def calculate_angle(a, b, c):

    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot = ba[0] * bc[0] + ba[1] * bc[1]

    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return 0

    cosine = dot / (mag_ba * mag_bc)
    cosine = max(-1, min(1, cosine))

    return math.degrees(math.acos(cosine))


def finger_is_open(landmarks, mcp, pip, tip):

    angle = calculate_angle(
        landmarks[mcp],
        landmarks[pip],
        landmarks[tip]
    )

    return angle > 155


def detect_gesture(landmarks):

    index = finger_is_open(landmarks, 5, 6, 8)
    middle = finger_is_open(landmarks, 9, 10, 12)
    ring = finger_is_open(landmarks, 13, 14, 16)
    pinky = finger_is_open(landmarks, 17, 18, 20)

    other_fingers_folded = (
        not index and not middle and not ring and not pinky
    )

    # THUMBS UP
    thumb_up = (
        landmarks[4].y < landmarks[3].y
        and landmarks[3].y < landmarks[2].y
    )

    if thumb_up and other_fingers_folded:
        return "THUMBS UP"

    # INDEX FINGER
    if index and not middle and not ring and not pinky:
        return "INDEX FINGER"

    # PEACE SIGN
    if index and middle and not ring and not pinky:
        return "PEACE"

    # OPEN PALM
    if index and middle and ring and pinky:
        return "OPEN PALM"

    # FIST
    if other_fingers_folded:
        return "FIST"

    return "UNKNOWN"


# -------------------------------
# PRESENTATION CONTROL
# -------------------------------

armed = True
last_action_time = 0


def control_powerpoint(gesture):

    global armed, last_action_time

    current_time = time.time()

    # FIST = RESET
    if gesture == "FIST":
        armed = True
        return

    # NO HAND = RESET
    if gesture == "NO HAND":
        armed = True
        return

    if not armed:
        return

    if current_time - last_action_time < COOLDOWN:
        return

    if gesture == "INDEX FINGER":

        pyautogui.press("right")
        print("NEXT SLIDE")

    elif gesture == "PEACE":

        pyautogui.press("left")
        print("PREVIOUS SLIDE")

    elif gesture == "OPEN PALM":

        pyautogui.press("f5")
        print("START PRESENTATION")

    elif gesture == "THUMBS UP":

        pyautogui.press("esc")
        print("END PRESENTATION")

    else:
        return

    last_action_time = current_time
    armed = False


# -------------------------------
# GESTURE STABILIZATION
# -------------------------------

gesture_history = deque(maxlen=3)
stable_gesture = "NO HAND"

# -------------------------------
# MAIN LOOP
# -------------------------------

start_time = time.perf_counter()

print("-----------------------------------")
print("AI SMART PRESENTATION ASSISTANT")
print("-----------------------------------")
print("INDEX       = NEXT SLIDE")
print("PEACE       = PREVIOUS SLIDE")
print("OPEN PALM   = START PRESENTATION")
print("FIST        = RESET / READY")
print("THUMBS UP   = END PRESENTATION")
print("Press Q to quit")
print("-----------------------------------")

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame not received.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        current_gesture = "NO HAND"

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            current_gesture = detect_gesture(landmarks)

            height, width, _ = frame.shape

            for landmark in landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

        # STABILIZE GESTURE
        gesture_history.append(current_gesture)

        if gesture_history.count(current_gesture) >= 2:
            stable_gesture = current_gesture

        # POWERPOINT CONTROL
        control_powerpoint(stable_gesture)

        # DISPLAY GESTURE
        cv2.putText(
            frame,
            "Gesture: " + stable_gesture,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # DISPLAY STATUS
        status = "READY" if armed else "MAKE A FIST TO RESET"

        cv2.putText(
            frame,
            "Control: " + status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "INDEX: NEXT | PEACE: PREVIOUS",
            (20, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "PALM: START | THUMB: END | FIST: RESET",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "AI Smart Presentation Assistant",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    detector.close()
    cv2.destroyAllWindows()

    print("Presentation Assistant closed.")
