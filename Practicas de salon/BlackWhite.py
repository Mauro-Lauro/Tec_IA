import numpy as np
import cv2
import os

# Initialize the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)
frame_counter = 0

# Create directories for storing images if they don't exist
if not os.path.exists('imgs'):
    os.makedirs('imgs')

if not os.path.exists('imgs/80'):
    os.makedirs('imgs/80')

if not os.path.exists('imgs/100'):
    os.makedirs('imgs/100')

if not os.path.exists('imgs/grayscale'):
    os.makedirs('imgs/grayscale')

if not os.path.exists('imgs/bw'):
    os.makedirs('imgs/bw')

def process_face_region(frame, x, y, w, h):
    """Extracts, resizes, and processes the detected face region."""
    face_region = frame[y:y+h, x:x+w]
    resized_face_80 = cv2.resize(face_region, (80, 80), interpolation=cv2.INTER_AREA)
    resized_face_100 = cv2.resize(face_region, (100, 100), interpolation=cv2.INTER_AREA)
    
    # Convert the resized face to grayscale (black and white)
    gray_face_80 = cv2.cvtColor(resized_face_80, cv2.COLOR_BGR2GRAY)
    gray_face_100 = cv2.cvtColor(resized_face_100, cv2.COLOR_BGR2GRAY)
    
    return resized_face_80, resized_face_100, gray_face_80, gray_face_100

def convert_to_black_and_white(gray_image):
    """Converts the grayscale image to black and white using a threshold."""
    _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return bw_image

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

        # Process the face region (resize and convert to grayscale)
        resized_face_80, resized_face_100, gray_face_80, gray_face_100 = process_face_region(frame, x, y, w, h)

        # Convert grayscale images to black and white
        bw_face_80 = convert_to_black_and_white(gray_face_80)
        bw_face_100 = convert_to_black_and_white(gray_face_100)

        # Save the images
        face_filename_80 = f'imgs/80/cara_{frame_counter}_80.jpg'
        face_filename_100 = f'imgs/100/cara_{frame_counter}_100.jpg'
        gray_face_filename_80 = f'imgs/grayscale/cara_{frame_counter}_80_gray.jpg'
        gray_face_filename_100 = f'imgs/grayscale/cara_{frame_counter}_100_gray.jpg'
        bw_face_filename_80 = f'imgs/bw/cara_{frame_counter}_80_bw.jpg'
        bw_face_filename_100 = f'imgs/bw/cara_{frame_counter}_100_bw.jpg'

        # Save the images
        cv2.imwrite(face_filename_80, resized_face_80)
        cv2.imwrite(face_filename_100, resized_face_100)
        cv2.imwrite(gray_face_filename_80, gray_face_80)
        cv2.imwrite(gray_face_filename_100, gray_face_100)
        cv2.imwrite(bw_face_filename_80, bw_face_80)
        cv2.imwrite(bw_face_filename_100, bw_face_100)

        # Display the processed face regions (optional)
        cv2.imshow('Resized Face 80x80', resized_face_80)
        cv2.imshow('Resized Face 100x100', resized_face_100)
        cv2.imshow('Grayscale Face 80x80', gray_face_80)
        cv2.imshow('Grayscale Face 100x100', gray_face_100)
        cv2.imshow('Black and White Face 80x80', bw_face_80)
        cv2.imshow('Black and White Face 100x100', bw_face_100)

    # Display the original frame with detected faces
    cv2.imshow('Detected Faces', frame)

    frame_counter += 1

    # Exit the loop when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
