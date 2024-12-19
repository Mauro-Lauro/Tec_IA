import numpy as np
import cv2
import math

# Initialize the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)
frame_counter = 0

def process_face_region(frame, x, y, w, h):
    """Extracts, resizes, and processes the detected face region."""
    face_region = frame[y:y+h, x:x+w]
    resized_face = cv2.resize(face_region, (80, 80), interpolation=cv2.INTER_AREA)
    gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
    return resized_face, gray_face

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Error capturing video. Exiting...")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Process the face region
        resized_face, gray_face = process_face_region(frame, x, y, w, h)

        # Display the processed face regions
        cv2.imshow('Resized Face', resized_face)
        cv2.imshow('Grayscale Face', gray_face)

    # Display the original frame with detected faces
    cv2.imshow('Detected Faces', frame)

    frame_counter += 1

    # Exit the loop when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
