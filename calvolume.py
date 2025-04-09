import cv2
import numpy as np
from collections import deque

# Configuration
VIDEO_PATH = r"E:\AASA IT SOLUTIONS\Quarry project\video\1.mp4"
REF_WIDTH_CM = 2.5  # Use a real small reference object (e.g., coin or sticker)
CALIBRATION_FRAMES = 50
MEASUREMENT_FRAMES = 60

# Buffers
calibration_buffer = deque(maxlen=CALIBRATION_FRAMES)
measurements = {
    'length': deque(maxlen=MEASUREMENT_FRAMES),
    'width': deque(maxlen=MEASUREMENT_FRAMES),
    'height': deque(maxlen=MEASUREMENT_FRAMES)
}

def get_stable_measurement(buffer):
    return np.median(buffer) if buffer else None

cap = cv2.VideoCapture(VIDEO_PATH)

try:
    if not cap.isOpened():
        raise ValueError("Cannot open video file - check path exists")

    # Save first frame for debugging
    ret, debug_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read video frames")
    cv2.imwrite("debug_frame.jpg", debug_frame)
    print("Saved debug_frame.jpg - verify reference object visibility")

    print("Starting calibration...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    while len(calibration_buffer) < CALIBRATION_FRAMES and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 100])
        upper = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / h

            if (0.5 < aspect_ratio < 2.0 and
                300 < area < 15000 and
                w > 20 and h > 20 and
                cv2.contourArea(cnt)/(w*h) > 0.5):
                valid_contours.append((area, w, cnt))

        if valid_contours:
            valid_contours.sort(reverse=True)
            _, w, cnt = valid_contours[0]
            calibration_buffer.append(w / REF_WIDTH_CM)
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
            cv2.putText(frame, f"Calibrating: {len(calibration_buffer)}/{CALIBRATION_FRAMES}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Calibration View', frame)
        cv2.imshow('Mask', cleaned)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if not calibration_buffer:
        raise ValueError("Calibration failed - check reference object visibility")

    pixels_per_cm = np.median(calibration_buffer)
    print(f"\nCalibration success: {pixels_per_cm:.2f} pixels/cm")
    print(f"Sanity check: 10cm should equal {10 * pixels_per_cm:.1f} pixels")

    print("\nStarting volume measurement...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(main_contour)
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            w_px = max(rect[1])
            h_px = min(rect[1])
            angle = abs(rect[2])

            if angle < 10 or angle > 80:  # Top-down view
                measurements['length'].append(w_px / pixels_per_cm)
                measurements['width'].append(h_px / pixels_per_cm)
                view_type = "TOP"
            else:  # Side view
                measurements['height'].append(h_px / pixels_per_cm)
                view_type = "SIDE"

            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            cv2.putText(frame, view_type, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Measurement', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    final_length = get_stable_measurement(measurements['length'])
    final_width = get_stable_measurement(measurements['width'])
    final_height = get_stable_measurement(measurements['height'])

    if None in (final_length, final_width, final_height):
        raise ValueError("Missing dimensions - show both top and side views")

    volume = final_length * final_width * final_height
    print(f"\n=== MEASUREMENT RESULTS ===")
    print(f"Length: {final_length:.1f} cm")
    print(f"Width: {final_width:.1f} cm")
    print(f"Height: {final_height:.1f} cm")
    print(f"Volume: {volume:.1f} cm³")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    print("Troubleshooting steps:")
    print("1. Check debug_frame.jpg for reference object visibility")
    print("2. Adjust HSV values in calibration phase")
    print("3. Verify camera focus and lighting")
finally:
    cap.release()
    cv2.destroyAllWindows()
