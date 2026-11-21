import cv2
import mediapipe as mp
import time
import math
import os
import json
import queue
import threading
import pyautogui
import sounddevice as sd

from collections import deque
from vosk import Model, KaldiRecognizer, SetLogLevel


# =========================================================
# 1. SETTINGS
# =========================================================

HAND_MODEL = os.path.join("models", "hand_landmarker.task")

VOICE_MODEL = os.path.join(
    "models",
    "vosk-model-small-en-us-0.15"
)

SAMPLE_RATE = 16000

GESTURE_COOLDOWN = 0.7
VOICE_DUPLICATE_COOLDOWN = 2.5

pyautogui.PAUSE = 0.02
SetLogLevel(-1)


# =========================================================
# 2. CHECK MODEL FILES
# =========================================================

if not os.path.exists(HAND_MODEL):
    print("Hand model not found:", HAND_MODEL)
    raise SystemExit

if not os.path.exists(VOICE_MODEL):
    print("Voice model not found:", VOICE_MODEL)
    raise SystemExit


# =========================================================
# 3. MEDIAPIPE INITIALIZATION
# =========================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(
    hand_options
)


# =========================================================
# 4. VOICE INITIALIZATION
# =========================================================

print("Loading voice model...")

voice_model = Model(VOICE_MODEL)

VOICE_COMMANDS = [
    "meow",
    "woof",
    "abracadabra",
    "freeze",
    "[unk]"
]

recognizer = KaldiRecognizer(
    voice_model,
    SAMPLE_RATE,
    json.dumps(VOICE_COMMANDS)
)

audio_queue = queue.Queue(maxsize=50)
action_queue = queue.Queue(maxsize=10)

stop_event = threading.Event()

last_voice_command_time = 0.0
last_voice_action = None


# =========================================================
# 5. MICROPHONE CALLBACK
# =========================================================

def audio_callback(indata, frames, time_info, status):

    if status:
        print("Microphone status:", status)

    try:
        audio_queue.put_nowait(bytes(indata))

    except queue.Full:
        pass


# =========================================================
# 6. VOICE COMMAND PROCESSING
# =========================================================

def process_voice_command(spoken_text):

    global last_voice_command_time, last_voice_action

    spoken_text = spoken_text.lower().strip()

    if not spoken_text:
        return

    print("VOICE HEARD:", spoken_text)

    command_map = {
        "meow": "NEXT",
        "woof": "PREVIOUS",
        "abracadabra": "START",
        "freeze": "STOP"
    }

    action = command_map.get(spoken_text)

    if action is None:
        print("Not a presentation command.")
        return

    current_time = time.monotonic()

    # Prevent duplicate recognition of the same command
    if (
        action == last_voice_action
        and current_time - last_voice_command_time
        < VOICE_DUPLICATE_COOLDOWN
    ):
        print("Repeated voice command ignored:", spoken_text)
        return

    last_voice_action = action
    last_voice_command_time = current_time

    try:
        action_queue.put_nowait(action)
        print("VOICE ACTION QUEUED:", action)

    except queue.Full:
        print("Action queue is full. Command ignored.")


# =========================================================
# 7. VOICE LISTENER THREAD
# =========================================================

def voice_listener():

    try:

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=audio_callback
        ):

            print("MICROPHONE ON!")
            print("Listening for voice commands...")

            while not stop_event.is_set():

                try:
                    audio_data = audio_queue.get(timeout=0.2)

                except queue.Empty:
                    continue

                if recognizer.AcceptWaveform(audio_data):

                    result = json.loads(recognizer.Result())

                    spoken_text = result.get("text", "")

                    if spoken_text:
                        process_voice_command(spoken_text)

    except Exception as error:

        print("VOICE ERROR:", error)


# =========================================================
# 8. FINGER DETECTION
# =========================================================

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

    other_fingers_folded = not any([
        index, middle, ring, pinky
    ])

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


# =========================================================
# 9. POWERPOINT WINDOW FOCUS
# =========================================================

def focus_powerpoint(action):

    try:

        windows = pyautogui.getWindowsWithTitle("PowerPoint")

        if not windows:

            print("PowerPoint window not found.")
            print("Please open PowerPoint.")

            return False

        slideshow_windows = [
            window for window in windows
            if "slide show" in window.title.lower()
        ]

        normal_windows = [
            window for window in windows
            if "slide show" not in window.title.lower()
        ]

        if action == "START":
            preferred_windows = normal_windows or windows

        else:
            preferred_windows = slideshow_windows or windows

        powerpoint_window = preferred_windows[0]

        if powerpoint_window.isMinimized:
            powerpoint_window.restore()

        powerpoint_window.activate()

        time.sleep(0.2)

        return True

    except Exception as error:

        print("PowerPoint focus warning:", error)

        return False


# =========================================================
# 10. SHARED ACTION CONTROLLER
# =========================================================

def execute_action(action):

    focused = focus_powerpoint(action)

    if not focused:

        print("Action cancelled. Please open PowerPoint.")

        return

    if action == "NEXT":

        pyautogui.press("right")
        print("ACTION: NEXT SLIDE")

    elif action == "PREVIOUS":

        pyautogui.press("left")
        print("ACTION: PREVIOUS SLIDE")

    elif action == "START":

        pyautogui.press("f5")
        print("ACTION: START PRESENTATION")

    elif action == "STOP":

        pyautogui.press("esc")
        print("ACTION: END PRESENTATION")


# =========================================================
# 11. GESTURE CONTROL
# =========================================================

gesture_armed = True
last_gesture_action_time = 0.0


def control_gesture(gesture):

    global gesture_armed, last_gesture_action_time

    current_time = time.monotonic()

    # FIST = RESET
    if gesture == "FIST":

        gesture_armed = True
        return

    # NO HAND = RESET
    if gesture == "NO HAND":

        gesture_armed = True
        return

    if not gesture_armed:
        return

    if current_time - last_gesture_action_time < GESTURE_COOLDOWN:
        return

    action = None

    if gesture == "INDEX FINGER":
        action = "NEXT"

    elif gesture == "PEACE":
        action = "PREVIOUS"

    elif gesture == "OPEN PALM":
        action = "START"

    elif gesture == "THUMBS UP":
        action = "STOP"

    if action:

        execute_action(action)

        last_gesture_action_time = time.monotonic()

        gesture_armed = False


# =========================================================
# 12. CAMERA INITIALIZATION
# =========================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():

    print("Camera could not be opened!")

    detector.close()

    raise SystemExit


# =========================================================
# 13. START VOICE THREAD
# =========================================================

voice_thread = threading.Thread(
    target=voice_listener,
    daemon=True
)

voice_thread.start()


# =========================================================
# 14. MAIN LOOP
# =========================================================

gesture_history = deque(maxlen=3)

stable_gesture = "NO HAND"

start_time = time.perf_counter()

print("-----------------------------------")
print("AI SMART PRESENTATION ASSISTANT")
print("GESTURE + VOICE CONTROL")
print("-----------------------------------")

print("GESTURE COMMANDS:")
print("INDEX FINGER = NEXT SLIDE")
print("PEACE       = PREVIOUS SLIDE")
print("OPEN PALM   = START")
print("FIST        = RESET")
print("THUMBS UP   = STOP")

print("-----------------------------------")

print("VOICE COMMANDS:")
print("MEOW        = NEXT SLIDE")
print("WOOF        = PREVIOUS SLIDE")
print("ABRACADABRA = START PRESENTATION")
print("FREEZE      = STOP PRESENTATION")

print("-----------------------------------")
print("Press Q in the camera window to quit.")


# =========================================================
# 15. MAIN CAMERA LOOP
# =========================================================

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

        # HANDLE GESTURE
        control_gesture(stable_gesture)

        # HANDLE VOICE ACTIONS
        while True:

            try:
                voice_action = action_queue.get_nowait()

            except queue.Empty:
                break

            execute_action(voice_action)

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

        status = "READY" if gesture_armed else "MAKE A FIST TO RESET"

        cv2.putText(
            frame,
            "Status: " + status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "VOICE CONTROL: ACTIVE",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SAY: MEOW | WOOF | ABRACADABRA | FREEZE",
            (20, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "INDEX: NEXT | PEACE: PREVIOUS | PALM: START",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "AI Smart Presentation Assistant",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# =========================================================
# 16. CLEANUP
# =========================================================

finally:

    print("Closing assistant...")

    stop_event.set()

    camera.release()

    detector.close()

    cv2.destroyAllWindows()

    print("Assistant closed successfully!")