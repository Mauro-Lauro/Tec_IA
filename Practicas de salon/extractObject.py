import cv2
import numpy as np
import os
import time

# Ensure the save directory exists
save_dir = 'assets/termo/'
os.makedirs(save_dir, exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)

# Define red color ranges in HSV (detect both red hues in HSV)
lower_red_1 = np.array([0, 120, 70])
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([170, 120, 70])
upper_red_2 = np.array([180, 255, 255])

# Initialize the counter for saved images
count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for red color in two ranges
    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
    # Combine the two red masks
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

    # Apply morphological transformations to clean up the mask
    mask_red = cv2.erode(mask_red, None, iterations=2)
    mask_red = cv2.dilate(mask_red, None, iterations=2)

    # Find contours in the red mask
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process the largest contour if it exists
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Draw and process if the radius is large enough
        if radius > 20:  # Filter out small contours
            top_left = (max(0, int(x - radius)), max(0, int(y - radius)))
            bottom_right = (min(frame.shape[1], int(x + radius)), min(frame.shape[0], int(y + radius)))

            if radius >= 50:  # Process only larger objects
                # Extract the region of interest (ROI)
                roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                if roi.size > 0:  # Ensure ROI is valid
                    # Resize the image
                    roi_resized = cv2.resize(roi, (100, 100))

                    # Display and save the image
                    cv2.imshow('Extracted Red Object', roi_resized)
                    count += 1
                    filename = os.path.join(save_dir, f'termo_{count}.jpg')
                    cv2.imwrite(filename, roi_resized)
                    print(f"Saved: {filename}")

                    # Wait briefly before capturing the next object
                    time.sleep(0.5)

    # Show the frames and mask
    cv2.imshow('Frame', frame)
    cv2.imshow('Red Mask', mask_red)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
