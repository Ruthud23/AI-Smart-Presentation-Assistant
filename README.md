# AI Smart Presentation Assistant 🎤🖐️

An intelligent, hands-free presentation control system that allows users to operate Microsoft PowerPoint using **hand gestures and voice commands**.

## 🚀 Project Overview

The AI Smart Presentation Assistant combines computer vision and speech recognition to make presentations easier and more interactive.

It uses a webcam to recognize hand gestures and a microphone to detect voice commands. These inputs are translated into keyboard actions to control PowerPoint slides.

## ✨ Key Features

- **Hand Gesture Recognition:** Control presentations using hand movements.
- **Voice Control:** Navigate slides using custom voice commands.
- **Live Camera Preview:** View hand detection through the dashboard.
- **Interactive Dashboard:** Monitor gestures, voice input and presentation actions.
- **Hands-Free Operation:** Start, navigate and stop slides without manually using a keyboard.

## 🖐️ Hand Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ Index Finger | Next Slide |
| ✌️ Peace Sign | Previous Slide |
| 🖐️ Open Palm | Start Slideshow |
| ✊ Fist | Reset / Ready |
| 👍 Thumbs Up | Stop Slideshow |

## 🎤 Voice Commands

| Voice Command | Action |
|---|---|
| Meow | Next Slide |
| Woof | Previous Slide |
| Abracadabra | Start Slideshow |
| Freeze | Stop Slideshow |

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- Vosk Speech Recognition
- CustomTkinter
- PyAutoGUI
- NumPy
- Pillow
- SoundDevice

## 📁 Project Structure

```text
AI-Smart-Presentation-Assistant/
│
├── dashboard.py
├── combined_assistant.py
├── gesture_recognition.py
├── hand_detection.py
├── voice_control.py
├── main.py
│
├── models/
│   ├── hand_landmarker.task
│   └── vosk-model-small-en-us-0.15/
│
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ruthud23/AI-Smart-Presentation-Assistant.git
```

### 2. Open the project folder

```bash
cd AI-Smart-Presentation-Assistant
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install opencv-python mediapipe pyautogui numpy customtkinter pillow sounddevice vosk pygetwindow
```

### 6. Download the required models

Place the MediaPipe hand landmarker model and the Vosk English speech model in the `models` folder using the paths expected by the application.

### 7. Run the application

```bash
python dashboard.py
```

Allow webcam and microphone access when prompted. Open your PowerPoint presentation to begin testing.

## 🎯 Applications

- Classroom presentations
- Business meetings
- Academic demonstrations
- Touch-free presentation environments
- Interactive presentations

## 🔮 Future Enhancements

- Support for additional presentation software.
- More customizable hand gestures.
- Improved speech recognition.
- Multi-language voice commands.
- Presentation analytics and gesture statistics.

## 👩‍💻 Developer

**Ruthu Devaraj Poojary**

GitHub: [Ruthud23](https://github.com/Ruthud23)

LinkedIn: [Ruthu Devaraj Poojary](https://linkedin.com/in/ruthu-devaraj-poojary-20230025a)

---

*Built with Python, Computer Vision and Speech Recognition.*