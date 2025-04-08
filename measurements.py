import cv2
import numpy as np

# Load video and reference object dimensions
video_path = r"C:\Users\Sahan\OneDrive\Desktop\workspace\router.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

ref_width_cm = 5.0  # Known width of reference object in centimeters
pixels_per_cm = None  # Will be calculated from reference object

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        continue
    
    # Sort contours by area (descending) and keep the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Find reference object (assuming it's the second largest)
    for i, cnt in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        # Calculate pixels_per_cm from reference object (assuming it's the second largest)
        if i == 1 and pixels_per_cm is None:
            pixels_per_cm = w / ref_width_cm
            print(f"Calibration: {pixels_per_cm:.2f} pixels per cm")
        
        # Draw contours and measurements
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        
        if pixels_per_cm is not None and i == 0:  # Measure main object (largest contour)
            length_cm = w / pixels_per_cm
            width_cm = h / pixels_per_cm
            
            # Display measurements
            cv2.putText(frame, f"L: {length_cm:.1f}cm", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"W: {width_cm:.1f}cm", (x, y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show the processed frame
    cv2.imshow('Measurement', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()