
import os
import json
import queue
import sounddevice as sd
import pyautogui

from vosk import Model, KaldiRecognizer, SetLogLevel

# -------------------------------
# SETTINGS
# -------------------------------

MODEL_PATH = os.path.join(
    "models",
    "vosk-model-small-en-us-0.15"
)

SAMPLE_RATE = 16000

pyautogui.PAUSE = 0.02
SetLogLevel(-1)

# -------------------------------
# COMMAND GRAMMAR
# -------------------------------

COMMANDS = [
    "next",
    "next slide",
    "go next",
    "move next",
    "previous",
    "previous slide",
    "go back",
    "back",
    "start",
    "start presentation",
    "begin presentation",
    "stop",
    "stop presentation",
    "end presentation",
    "quit voice control",
    "exit voice control",
    "[unk]"
]

# -------------------------------
# LOAD MODEL
# -------------------------------

if not os.path.exists(MODEL_PATH):
    print("Voice model not found!")
    raise SystemExit

print("Loading voice model...")

model = Model(MODEL_PATH)

recognizer = KaldiRecognizer(
    model,
    SAMPLE_RATE,
    json.dumps(COMMANDS)
)

audio_queue = queue.Queue()

# -------------------------------
# MICROPHONE CALLBACK
# -------------------------------

def audio_callback(indata, frames, time_info, status):

    if status:
        print("Microphone status:", status)

    audio_queue.put(bytes(indata))


# -------------------------------
# PROCESS COMMAND
# -------------------------------

def process_command(text):

    text = text.lower().strip()

    if not text:
        return True

    print("You said:", text)

    words = set(text.split())

    # EXIT VOICE CONTROL
    if "quit" in words or "exit" in words:
        print("Closing voice control...")
        return False

    # NEXT SLIDE
    if "next" in words:
        pyautogui.press("right")
        print("ACTION: NEXT SLIDE")

    # PREVIOUS SLIDE
    elif "previous" in words or "back" in words:
        pyautogui.press("left")
        print("ACTION: PREVIOUS SLIDE")

    # START PRESENTATION
    elif "start" in words or "begin" in words:
        pyautogui.press("f5")
        print("ACTION: START PRESENTATION")

    # STOP PRESENTATION
    elif "stop" in words or "end" in words:
        pyautogui.press("esc")
        print("ACTION: STOP PRESENTATION")

    else:
        print("Command not recognized. Please try again.")

    return True


# -------------------------------
# MAIN PROGRAM
# -------------------------------

print("--------------------------------")
print("AI SMART PRESENTATION ASSISTANT")
print("VOICE CONTROL")
print("--------------------------------")

print("Available commands:")
print("Next slide")
print("Previous slide")
print("Start presentation")
print("Stop presentation")
print("Quit voice control")
print("--------------------------------")

try:

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):

        print("MICROPHONE ON!")
        print("Listening for commands...")

        running = True

        while running:

            data = audio_queue.get()

            if recognizer.AcceptWaveform(data):

                result = json.loads(recognizer.Result())

                spoken_text = result.get("text", "")

                if spoken_text:
                    running = process_command(spoken_text)

except KeyboardInterrupt:

    print("\nVoice control stopped.")

except Exception as error:

    print("ERROR:", error)

finally:

    print("Voice control closed.")
