
import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------------------
# 1. LOAD MEDIAPIPE MODEL
# -------------------------------

model_path = "models/hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(options)


# -------------------------------
# 2. OPEN WEBCAM
# -------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam not found!")
    detector.close()
    exit()

print("Hand Detection Started!")
print("Press Q to exit.")


# -------------------------------
# 3. FINGER DETECTION FUNCTION
# -------------------------------

def detect_fingers(hand, handedness):

    fingers = []

    # Thumb detection using distance between landmarks
    thumb_tip = hand[4]
    thumb_ip = hand[3]
    thumb_mcp = hand[2]
    index_mcp = hand[5]

    # Compare thumb tip's distance from index MCP
    tip_distance = math.hypot(
        thumb_tip.x - index_mcp.x,
        thumb_tip.y - index_mcp.y
    )

    joint_distance = math.hypot(
        thumb_ip.x - index_mcp.x,
        thumb_ip.y - index_mcp.y
    )

    thumb_open = tip_distance > joint_distance + 0.04

    fingers.append(thumb_open)

    # Index, Middle, Ring and Pinky
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):

        is_open = hand[tip].y < hand[pip].y

        fingers.append(is_open)

    return fingers


# -------------------------------
# 4. MAIN WEBCAM LOOP
# -------------------------------

finger_names = [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Pinky"
]

try:

    while True:

        success, frame = cap.read()

        if not success:
            print("Unable to read webcam!")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands
        result = detector.detect(mp_image)

        if result.hand_landmarks:

            for hand_index, hand in enumerate(result.hand_landmarks):

                handedness = result.handedness[hand_index][0].category_name

                # Draw landmarks
                for landmark in hand:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Detect fingers
                fingers = detect_fingers(hand, handedness)

                # Display hand information
                cv2.putText(
                    frame,
                    f"{handedness} HAND",
                    (10, 35 + hand_index * 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                for i, name in enumerate(finger_names):

                    status = "OPEN" if fingers[i] else "CLOSED"

                    color = (
                        (0, 255, 0)
                        if fingers[i]
                        else (0, 0, 255)
                    )

                    cv2.putText(
                        frame,
                        f"{name}: {status}",
                        (10, 75 + hand_index * 240 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Show webcam
        cv2.imshow(
            "AI Smart Presentation - Hand Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    print("Hand Detection Stopped!")
