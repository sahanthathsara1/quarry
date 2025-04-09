import cv2
import numpy as np

# Load video and reference object dimensions
video_path = r"C:\Users\Sahan\OneDrive\Desktop\workspace\router.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Measurement variables
ref_width_cm = 5.0  # Known width of reference object in centimeters
pixels_per_cm = None
measurements = {'length': None, 'width': None, 'height': None}

def calculate_volume():
    if None in measurements.values():
        return None
    return measurements['length'] * measurements['width'] * measurements['height']

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
    
    # Sort contours by area (descending)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Detect reference object (second largest contour)
    if len(contours) > 1:
        (x_ref, y_ref, w_ref, h_ref) = cv2.boundingRect(contours[1])
        if pixels_per_cm is None:
            pixels_per_cm = w_ref / ref_width_cm
            print(f"Calibration: {pixels_per_cm:.2f} pixels per cm")
    
    if pixels_per_cm and len(contours) > 0:
        # Main object (largest contour)
        (x, y, w, h) = cv2.boundingRect(contours[0])
        
        # Determine measurement type based on aspect ratio
        aspect_ratio = w / h
        
        if 0.9 < aspect_ratio < 1.1:  # Top view (square aspect)
            measurements['length'] = w / pixels_per_cm
            measurements['width'] = h / pixels_per_cm
        else:  # Side view (rectangular aspect)
            measurements['height'] = h / pixels_per_cm
        
        # Draw measurements
        cv2.drawContours(frame, [contours[0]], -1, (0, 255, 0), 2)
        text_y = y - 20
        for dim, value in measurements.items():
            if value is not None:
                cv2.putText(frame, f"{dim}: {value:.1f}cm", (x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                text_y -= 30
        
        # Calculate and display volume
        volume = calculate_volume()
        if volume:
            cv2.putText(frame, f"Volume: {volume:.1f}cm³", (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            print(f"Volume: {volume:.1f}cm^3")
    
    # Show the processed frame
    cv2.imshow('Volume Measurement', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
# cv2.destroyAllWindows()
