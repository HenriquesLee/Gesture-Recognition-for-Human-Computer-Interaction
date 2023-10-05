import cv2
import numpy as np

# Load the pre-trained Haarcascades for face and hand detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')

# Open a connection to the camera (0 indicates the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert the frame to HSV for hand detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Bitwise-AND between the mask and the frame
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(res, 1.3, 5)

    # Draw rectangles around the hands
    for (hx, hy, hw, hh) in hands:
        cv2.rectangle(frame, (hx, hy), (hx+hw, hy+hh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
