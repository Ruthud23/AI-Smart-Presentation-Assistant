
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera:", cap.isOpened())

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    cv2.imshow("LIVE CAMERA TEST", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
