import cv2

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Initialize webcam
video_capture = cv2.VideoCapture(0)

def draw_face_features(image, x, y, w, h):
    """Draws rectangles and circles to create facial features."""
    # Draw face rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), (234, 23, 23), 2)
    # Draw mouth rectangle
    cv2.rectangle(image, (x, y + h // 2), (x + w, y + h), (0, 255, 0), 5)
    # Draw eyes
    eye_radius = 20
    pupil_radius = 5
    left_eye_center = (x + int(w * 0.3), y + int(h * 0.4))
    right_eye_center = (x + int(w * 0.7), y + int(h * 0.4))
    
    for eye_center in [left_eye_center, right_eye_center]:
        cv2.circle(image, eye_center, eye_radius, (255, 255, 255), -1)  # White of the eye
        cv2.circle(image, eye_center, pupil_radius, (0, 0, 255), -1)   # Pupil

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    # Draw features for each detected face
    for (x, y, w, h) in faces:
        draw_face_features(frame, x, y, w, h)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
