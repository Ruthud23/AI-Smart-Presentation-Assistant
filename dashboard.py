
import os
import time
import math
import json
import queue
import threading
import ctypes
import tkinter as tk
from collections import deque

import cv2
import mediapipe as mp
import customtkinter as ctk
import pyautogui
import pygetwindow as gw
import sounddevice as sd

from PIL import Image, ImageTk
from vosk import Model, KaldiRecognizer, SetLogLevel

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =====================================================
# SETTINGS
# =====================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

SetLogLevel(-1)

APP_TITLE = "AI Smart Presentation Assistant"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HAND_MODEL = os.path.join(
    BASE_DIR, "models", "hand_landmarker.task"
)

VOICE_MODEL = os.path.join(
    BASE_DIR, "models", "vosk-model-small-en-us-0.15"
)

SAMPLE_RATE = 16000
GESTURE_COOLDOWN = 0.7
VOICE_DUPLICATE_COOLDOWN = 2.5

VOICE_COMMANDS = [
    "meow", "woof", "abracadabra", "freeze", "[unk]"
]

VOICE_MAP = {
    "meow": "NEXT",
    "woof": "PREVIOUS",
    "abracadabra": "START",
    "freeze": "STOP"
}


# =====================================================
# APPLICATION
# =====================================================

class PresentationAssistant(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1250x850")
        self.minsize(1100, 760)
        self.configure(fg_color="#10121C")

        self.cap = None
        self.detector = None
        self.camera_running = False
        self.timestamp = 0

        self.history = deque(maxlen=3)
        self.stable_gesture = "NO HAND"
        self.gesture_armed = True
        self.last_gesture_action_time = 0

        self.voice_model = None
        self.voice_thread = None
        self.voice_running = False
        self.voice_stop_event = threading.Event()

        self.audio_queue = queue.Queue(maxsize=50)
        self.action_queue = queue.Queue(maxsize=10)

        self.last_voice_action = None
        self.last_voice_command_time = 0

        self.total_actions = 0
        self.next_count = 0
        self.previous_count = 0

        self.current_action = "Waiting for action"

        self.preview_image = None

        self.build_dashboard()

        self.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

    # =================================================
    # DASHBOARD DESIGN
    # =================================================

    def build_dashboard(self):

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(
            self, fg_color="#191C2B",
            corner_radius=15, height=80
        )
        header.grid(
            row=0, column=0, columnspan=2,
            padx=18, pady=15, sticky="ew"
        )
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="AI SMART PRESENTATION ASSISTANT",
            font=("Arial", 23, "bold"),
            text_color="#FFFFFF"
        ).grid(
            row=0, column=0, padx=20,
            pady=(12, 0), sticky="w"
        )

        ctk.CTkLabel(
            header,
            text="Gesture + Voice Controlled Presentation System",
            font=("Arial", 12),
            text_color="#9A9EB5"
        ).grid(
            row=1, column=0, padx=20,
            pady=(0, 12), sticky="w"
        )

        self.connection_label = ctk.CTkLabel(
            header,
            text="● CAMERA OFF",
            text_color="#F87171",
            font=("Arial", 13, "bold")
        )
        self.connection_label.grid(
            row=0, column=1, rowspan=2, padx=25
        )

        # LEFT PANEL

        left = ctk.CTkFrame(
            self, fg_color="#191C2B", corner_radius=15
        )
        left.grid(
            row=1, column=0,
            padx=(18, 9), pady=(0, 18), sticky="nsew"
        )

        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            left,
            text="LIVE CAMERA FEED",
            font=("Arial", 17, "bold")
        ).grid(
            row=0, column=0, padx=20,
            pady=15, sticky="w"
        )

        # STANDARD TKINTER LABEL FOR CAMERA PREVIEW

        self.camera_label = tk.Label(
            left,
            text="CAMERA NOT STARTED\n\nClick START ASSISTANT",
            font=("Arial", 18, "bold"),
            fg="#8B91A8",
            bg="#0B0D15",
            bd=0,
            highlightthickness=0,
            anchor="center",
            justify="center"
        )

        self.camera_label.grid(
            row=1, column=0,
            padx=20, pady=5, sticky="nsew"
        )

        gesture_frame = ctk.CTkFrame(
            left, fg_color="#22263A", corner_radius=12
        )
        gesture_frame.grid(
            row=2, column=0,
            padx=20, pady=12, sticky="ew"
        )

        ctk.CTkLabel(
            gesture_frame,
            text="CURRENT GESTURE",
            font=("Arial", 12, "bold"),
            text_color="#9A9EB5"
        ).pack(pady=(10, 3))

        self.gesture_label = ctk.CTkLabel(
            gesture_frame,
            text="NO HAND",
            font=("Arial", 25, "bold"),
            text_color="#38BDF8"
        )
        self.gesture_label.pack(pady=(0, 10))

        camera_buttons = ctk.CTkFrame(
            left, fg_color="transparent"
        )
        camera_buttons.grid(
            row=3, column=0,
            padx=20, pady=(0, 15), sticky="ew"
        )

        camera_buttons.grid_columnconfigure(
            (0, 1), weight=1
        )

        self.start_button = ctk.CTkButton(
            camera_buttons,
            text="▶ START ASSISTANT",
            height=42,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            command=self.start_camera
        )
        self.start_button.grid(
            row=0, column=0, padx=5, sticky="ew"
        )

        self.stop_button = ctk.CTkButton(
            camera_buttons,
            text="■ STOP ASSISTANT",
            height=42,
            fg_color="#991B1B",
            hover_color="#7F1D1D",
            command=self.stop_camera
        )
        self.stop_button.grid(
            row=0, column=1, padx=5, sticky="ew"
        )

        # RIGHT PANEL

        right = ctk.CTkFrame(
            self, fg_color="#191C2B", corner_radius=15
        )
        right.grid(
            row=1, column=1,
            padx=(9, 18), pady=(0, 18), sticky="nsew"
        )

        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right,
            text="FINGER DETECTION",
            font=("Arial", 17, "bold")
        ).grid(
            row=0, column=0, padx=20,
            pady=12, sticky="w"
        )

        self.finger_labels = {}

        fingers = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]

        for i, finger in enumerate(fingers):

            card = ctk.CTkFrame(
                right, fg_color="#252A3D",
                corner_radius=9, height=38
            )
            card.grid(
                row=i + 1, column=0,
                padx=20, pady=3, sticky="ew"
            )

            card.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                card,
                text=finger,
                font=("Arial", 12, "bold")
            ).grid(
                row=0, column=0, padx=12,
                pady=7, sticky="w"
            )

            status = ctk.CTkLabel(
                card,
                text="OFF",
                text_color="#F87171",
                font=("Arial", 12, "bold")
            )

            status.grid(row=0, column=1, padx=12)

            self.finger_labels[finger] = status

        # POWERPOINT CONTROLS

        ctk.CTkLabel(
            right,
            text="POWERPOINT CONTROLS",
            font=("Arial", 16, "bold")
        ).grid(
            row=6, column=0, padx=20,
            pady=(12, 7), sticky="w"
        )

        controls = [
            ("NEXT SLIDE  →", "NEXT"),
            ("←  PREVIOUS SLIDE", "PREVIOUS"),
            ("START SLIDESHOW", "START"),
            ("STOP SLIDESHOW", "STOP")
        ]

        for i, (name, action) in enumerate(controls):

            button = ctk.CTkButton(
                right,
                text=name,
                height=34,
                fg_color="#353B57",
                hover_color="#4F46E5",
                command=lambda a=action: self.execute_action(a)
            )

            button.grid(
                row=7 + i, column=0,
                padx=20, pady=3, sticky="ew"
            )

        # VOICE CONTROL

        voice_frame = ctk.CTkFrame(
            right, fg_color="#22263A", corner_radius=12
        )

        voice_frame.grid(
            row=11, column=0,
            padx=20, pady=(10, 5), sticky="ew"
        )

        ctk.CTkLabel(
            voice_frame,
            text="VOICE CONTROL",
            font=("Arial", 13, "bold"),
            text_color="#9A9EB5"
        ).pack(pady=(8, 2))

        self.voice_status_label = ctk.CTkLabel(
            voice_frame,
            text="● VOICE OFF",
            font=("Arial", 12, "bold"),
            text_color="#F87171"
        )
        self.voice_status_label.pack(pady=2)

        ctk.CTkLabel(
            voice_frame,
            text=(
                "MEOW = Next   |   WOOF = Previous\n"
                "ABRACADABRA = Start   |   FREEZE = Stop"
            ),
            font=("Arial", 10),
            text_color="#D1D5DB",
            justify="center"
        ).pack(pady=(2, 8))

        # STATISTICS

        stats = ctk.CTkFrame(
            right, fg_color="#22263A", corner_radius=12
        )

        stats.grid(
            row=12, column=0,
            padx=20, pady=5, sticky="ew"
        )

        ctk.CTkLabel(
            stats,
            text="PRESENTATION STATISTICS",
            font=("Arial", 12, "bold"),
            text_color="#9A9EB5"
        ).pack(pady=(7, 3))

        self.stats_label = ctk.CTkLabel(
            stats,
            text="Total actions: 0\nNext: 0   Previous: 0",
            font=("Arial", 12),
            justify="center"
        )
        self.stats_label.pack(pady=(0, 7))

        self.action_label = ctk.CTkLabel(
            right,
            text="Waiting for action...",
            font=("Arial", 11),
            text_color="#38BDF8",
            wraplength=300
        )

        self.action_label.grid(
            row=13, column=0, padx=15, pady=(4, 10)
        )

    # =================================================
    # FINGER DETECTION
    # =================================================

    def calculate_angle(self, a, b, c):

        angle1 = math.atan2(a.y - b.y, a.x - b.x)
        angle2 = math.atan2(c.y - b.y, c.x - b.x)

        angle = abs(math.degrees(angle1 - angle2))

        if angle > 180:
            angle = 360 - angle

        return angle

    def finger_is_open(self, hand, mcp, pip, tip):

        angle = self.calculate_angle(
            hand[mcp], hand[pip], hand[tip]
        )

        return angle > 155

    def detect_fingers(self, hand):

        index = self.finger_is_open(hand, 5, 6, 8)
        middle = self.finger_is_open(hand, 9, 10, 12)
        ring = self.finger_is_open(hand, 13, 14, 16)
        pinky = self.finger_is_open(hand, 17, 18, 20)

        thumb = (
            hand[4].y < hand[3].y
            and hand[3].y < hand[2].y
        )

        return [thumb, index, middle, ring, pinky]

    # =================================================
    # GESTURE RECOGNITION
    # =================================================

    def recognize_gesture(self, fingers):

        thumb, index, middle, ring, pinky = fingers

        if thumb and not index and not middle and not ring and not pinky:
            return "THUMBS UP"

        if index and not middle and not ring and not pinky:
            return "INDEX FINGER"

        if index and middle and not ring and not pinky:
            return "PEACE"

        if index and middle and ring and pinky:
            return "OPEN PALM"

        if not index and not middle and not ring and not pinky:
            return "FIST"

        return "UNKNOWN"

    def update_finger_display(self, fingers):

        names = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]

        for name, is_open in zip(names, fingers):

            label = self.finger_labels[name]

            if is_open:
                label.configure(
                    text="OPEN", text_color="#4ADE80"
                )
            else:
                label.configure(
                    text="CLOSED", text_color="#F87171"
                )

    # =================================================
    # POWERPOINT WINDOW DETECTION
    # =================================================

    def find_powerpoint_window(self, action):

        windows = gw.getAllWindows()
        candidates = []

        for window in windows:

            title = (window.title or "").lower()

            if "powerpoint" in title:
                candidates.append(window)

        if not candidates:
            return None

        slideshow_windows = [
            w for w in candidates
            if "slide show" in w.title.lower()
        ]

        normal_windows = [
            w for w in candidates
            if "slide show" not in w.title.lower()
        ]

        if action == "START":
            preferred = normal_windows

            if not preferred:
                return None
        else:
            preferred = slideshow_windows or normal_windows

        for window in preferred:

            try:
                if window.isVisible and not window.isMinimized:
                    return window
            except Exception:
                continue

        return preferred[0] if preferred else None

    # =================================================
    # ACTIVATE POWERPOINT WINDOW
    # =================================================

    def focus_powerpoint(self, action):

        try:

            ppt_window = self.find_powerpoint_window(action)

            if ppt_window is None:
                print("POWERPOINT WINDOW NOT FOUND:", action)
                return False

            hwnd = int(ppt_window._hWnd)
            user32 = ctypes.windll.user32

            user32.ShowWindow(hwnd, 9)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)

            try:
                ppt_window.activate()
            except Exception as error:
                print("PyGetWindow activation message:", error)

            time.sleep(0.3)

            active_window = gw.getActiveWindow()

            if active_window:

                print("ACTIVE WINDOW:", active_window.title)

                if "powerpoint" not in active_window.title.lower():
                    print("PowerPoint did not receive focus.")
                    return False

            print("POWERPOINT FOCUSED:", ppt_window.title)

            return True

        except Exception as error:

            print("POWERPOINT FOCUS ERROR:", error)

            return False

    # =================================================
    # EXECUTE POWERPOINT ACTION
    # =================================================

    def execute_action(self, action):

        key_map = {
            "NEXT": "right",
            "PREVIOUS": "left",
            "START": "f5",
            "STOP": "esc"
        }

        if action not in key_map:
            return

        if not self.focus_powerpoint(action):

            if action == "START":
                message = (
                    "PowerPoint presentation window not found. "
                    "Exit slideshow first."
                )
            else:
                message = (
                    "PowerPoint slideshow not found. "
                    "Open your presentation first."
                )

            self.action_label.configure(
                text=message,
                text_color="#F87171"
            )

            return

        try:

            time.sleep(0.15)

            pyautogui.press(key_map[action])

            action_names = {
                "NEXT": "Next slide",
                "PREVIOUS": "Previous slide",
                "START": "Start slideshow",
                "STOP": "Stop slideshow"
            }

            self.current_action = action_names[action]

            self.action_label.configure(
                text="Action: " + self.current_action,
                text_color="#4ADE80"
            )

            self.total_actions += 1

            if action == "NEXT":
                self.next_count += 1
            elif action == "PREVIOUS":
                self.previous_count += 1

            self.update_statistics()

            print("POWERPOINT ACTION SENT:", action)

        except Exception as error:

            self.action_label.configure(
                text="Action error: " + str(error)[:80],
                text_color="#F87171"
            )

            print("ACTION ERROR:", error)

    # =================================================
    # GESTURE CONTROL
    # =================================================

    def control_gesture(self, gesture):

        current_time = time.monotonic()

        if gesture in ["FIST", "NO HAND"]:
            self.gesture_armed = True
            return

        if gesture == "UNKNOWN":
            return

        if not self.gesture_armed:
            return

        if current_time - self.last_gesture_action_time < GESTURE_COOLDOWN:
            return

        action_map = {
            "INDEX FINGER": "NEXT",
            "PEACE": "PREVIOUS",
            "OPEN PALM": "START",
            "THUMBS UP": "STOP"
        }

        action = action_map.get(gesture)

        if action:

            self.execute_action(action)

            self.last_gesture_action_time = time.monotonic()
            self.gesture_armed = False

    # =================================================
    # VOICE CONTROL
    # =================================================

    def audio_callback(self, indata, frames, time_info, status):

        if status:
            print("Microphone status:", status)

        try:
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def process_voice_command(self, spoken_text):

        spoken_text = spoken_text.lower().strip()

        if not spoken_text:
            return

        print("VOICE HEARD:", spoken_text)

        action = VOICE_MAP.get(spoken_text)

        if not action:
            return

        current_time = time.monotonic()

        if (
            action == self.last_voice_action
            and current_time - self.last_voice_command_time
            < VOICE_DUPLICATE_COOLDOWN
        ):

            print("Repeated voice command ignored.")
            return

        self.last_voice_action = action
        self.last_voice_command_time = current_time

        try:
            self.action_queue.put_nowait(action)
            print("VOICE ACTION QUEUED:", action)
        except queue.Full:
            print("Voice action queue is full.")

    def voice_listener(self):

        try:

            recognizer = KaldiRecognizer(
                self.voice_model,
                SAMPLE_RATE,
                json.dumps(VOICE_COMMANDS)
            )

            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self.audio_callback
            ):

                while not self.voice_stop_event.is_set():

                    try:
                        audio_data = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(audio_data):

                        result = json.loads(recognizer.Result())

                        spoken_text = result.get("text", "")

                        if spoken_text:
                            self.process_voice_command(spoken_text)

        except Exception as error:

            print("VOICE ERROR:", error)

            if not self.voice_stop_event.is_set():

                self.after(
                    0,
                    lambda: self.voice_status_label.configure(
                        text="● VOICE ERROR",
                        text_color="#F87171"
                    )
                )

    def start_voice_control(self):

        if self.voice_running:
            return

        if not os.path.exists(VOICE_MODEL):

            self.voice_status_label.configure(
                text="● VOICE MODEL MISSING",
                text_color="#F87171"
            )

            return

        try:

            if self.voice_model is None:
                self.voice_model = Model(VOICE_MODEL)

            self.voice_stop_event.clear()

            while not self.audio_queue.empty():

                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            self.voice_running = True

            self.voice_thread = threading.Thread(
                target=self.voice_listener,
                daemon=True
            )

            self.voice_thread.start()

            self.voice_status_label.configure(
                text="● VOICE STARTING...",
                text_color="#FACC15"
            )

            self.after(1000, self.update_voice_status)

        except Exception as error:

            self.voice_running = False

            self.voice_status_label.configure(
                text="● VOICE ERROR",
                text_color="#F87171"
            )

            print("VOICE START ERROR:", error)

    def update_voice_status(self):

        if self.voice_running and not self.voice_stop_event.is_set():

            self.voice_status_label.configure(
                text="● VOICE CONTROL ACTIVE",
                text_color="#4ADE80"
            )

    def stop_voice_control(self):

        self.voice_running = False
        self.voice_stop_event.set()

        self.voice_status_label.configure(
            text="● VOICE OFF",
            text_color="#F87171"
        )

    # =================================================
    # STATISTICS
    # =================================================

    def update_statistics(self):

        self.stats_label.configure(
            text=(
                f"Total actions: {self.total_actions}\n"
                f"Next: {self.next_count}   "
                f"Previous: {self.previous_count}"
            )
        )

    # =================================================
    # START ASSISTANT
    # =================================================

    def start_camera(self):

        if self.camera_running:
            return

        if not os.path.exists(HAND_MODEL):

            self.action_label.configure(
                text="Hand model missing!",
                text_color="#F87171"
            )

            return

        try:

            base_options = python.BaseOptions(
                model_asset_path=HAND_MODEL
            )

            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6
            )

            self.detector = vision.HandLandmarker.create_from_options(
                options
            )

            self.cap = cv2.VideoCapture(
                0, cv2.CAP_DSHOW
            )

            if not self.cap.isOpened():

                self.cap.release()
                self.cap = None

                self.detector.close()
                self.detector = None

                self.action_label.configure(
                    text="Camera not found!",
                    text_color="#F87171"
                )

                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.camera_running = True
            self.timestamp = 0

            self.history.clear()

            self.stable_gesture = "NO HAND"
            self.gesture_armed = True

            self.connection_label.configure(
                text="● CAMERA CONNECTED",
                text_color="#4ADE80"
            )

            self.action_label.configure(
                text="Assistant started!",
                text_color="#4ADE80"
            )

            self.start_voice_control()

            self.update_camera()

        except Exception as error:

            self.camera_running = False

            if self.cap is not None:
                self.cap.release()
                self.cap = None

            if self.detector is not None:
                self.detector.close()
                self.detector = None

            self.connection_label.configure(
                text="● CAMERA OFF",
                text_color="#F87171"
            )

            self.action_label.configure(
                text="Startup error: " + str(error)[:100],
                text_color="#F87171"
            )

            print("CAMERA START ERROR:", error)

    # =================================================
    # CAMERA PROCESSING
    # =================================================

    def update_camera(self):

        if not self.camera_running:
            return

        try:

            if self.cap is None or self.detector is None:
                raise RuntimeError(
                    "Camera or hand detector is unavailable."
                )

            success, frame = self.cap.read()

            if not success or frame is None:
                raise RuntimeError(
                    "Unable to read camera frame."
                )

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            self.timestamp += 33

            result = self.detector.detect_for_video(
                mp_image, self.timestamp
            )

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                fingers = self.detect_fingers(hand)

                gesture = self.recognize_gesture(fingers)

                self.history.append(gesture)

                self.update_finger_display(fingers)

                height, width, _ = frame.shape

                for point in hand:

                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

            else:

                fingers = [
                    False, False, False, False, False
                ]

                self.history.clear()

                gesture = "NO HAND"

                self.update_finger_display(fingers)

            # STABILIZE GESTURE

            if self.history:

                most_common = max(
                    set(self.history),
                    key=self.history.count
                )

                if self.history.count(most_common) >= 2:
                    self.stable_gesture = most_common

            else:
                self.stable_gesture = "NO HAND"

            self.control_gesture(
                self.stable_gesture
            )

            self.gesture_label.configure(
                text=self.stable_gesture
            )

            # PROCESS VOICE ACTIONS

            while True:

                try:
                    action = self.action_queue.get_nowait()
                except queue.Empty:
                    break

                self.execute_action(action)

            # -----------------------------------------
            # CAMERA DISPLAY - STANDARD TKINTER LABEL
            # -----------------------------------------

            display_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(display_frame)

            pil_image = pil_image.resize(
                (680, 430),
                Image.Resampling.LANCZOS
            )

            self.preview_image = ImageTk.PhotoImage(
                pil_image,
                master=self.camera_label
            )

            self.camera_label.configure(
                image=self.preview_image,
                text=""
            )

            self.camera_label.image = self.preview_image

            if self.camera_running:
                self.after(30, self.update_camera)

        except Exception as error:

            print(
                "CAMERA PREVIEW ERROR:",
                repr(error)
            )

            self.action_label.configure(
                text="Camera preview error: " + str(error)[:90],
                text_color="#F87171"
            )

            if self.camera_running:
                self.after(500, self.update_camera)

    # =================================================
    # STOP ASSISTANT
    # =================================================

    def stop_camera(self):

        self.camera_running = False

        self.stop_voice_control()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.detector is not None:
            self.detector.close()
            self.detector = None

        self.camera_label.configure(
            image="",
            text="ASSISTANT STOPPED",
            fg="#9A9EB5",
            bg="#0B0D15"
        )

        self.camera_label.image = None
        self.preview_image = None

        self.gesture_label.configure(
            text="NO HAND"
        )

        self.connection_label.configure(
            text="● CAMERA OFF",
            text_color="#F87171"
        )

        self.history.clear()
        self.stable_gesture = "NO HAND"
        self.gesture_armed = True

        self.update_finger_display(
            [False, False, False, False, False]
        )

        self.action_label.configure(
            text="Assistant stopped"
        )

    # =================================================
    # CLOSE APPLICATION
    # =================================================

    def close_application(self):

        self.stop_camera()
        self.destroy()


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app = PresentationAssistant()
    app.mainloop()
