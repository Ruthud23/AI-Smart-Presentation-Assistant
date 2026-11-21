
import cv2

# Step 1: Open the default webcam
camera = cv2.VideoCapture(0)

# Step 2: Check whether the webcam is available
if not camera.isOpened():
    print("Error: Webcam could not be accessed!")
    exit()

print("Webcam started successfully!")
print("Press Q to close the camera.")

# Step 3: Continuously capture video
while True:

    success, frame = camera.read()

    if not success:
        print("Unable to capture video.")
        break

    # Step 4: Display the video
    cv2.imshow("AI Smart Presentation Assistant", frame)

    # Step 5: Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 6: Release camera resources
camera.release()
cv2.destroyAllWindows()

print("Webcam closed successfully!")
