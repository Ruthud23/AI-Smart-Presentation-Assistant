
import cv2

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Camera started! Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    cv2.imshow("AI Smart Presentation Assistant - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")
