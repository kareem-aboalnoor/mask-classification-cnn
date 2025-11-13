import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load mask detection model
maskNet = load_model("mask_detector.h5")

# Open camera or video
cap = cv2.VideoCapture("34775-405202946_small.mp4")

# Set display window size
display_width = 1280
display_height = 720

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Keep original frame for display
    display_frame = frame.copy()
    
    # Prepare frame for model
    face = cv2.resize(frame, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    # Predict mask
    (mask, withoutMask) = maskNet.predict(face)[0]

    # Determine label, color and percentage
    if mask > withoutMask:
        label = "Mask"
        color = (0, 255, 0)  # Green
        percent = mask * 100
    else:
        label = "No Mask"
        color = (0, 0, 255)  # Red
        percent = withoutMask * 100

    # Resize frame for display
    display_frame = cv2.resize(display_frame, (display_width, display_height))
    
    # Calculate text size
    font_scale = display_width / 1000
    thickness = int(display_width / 400)
    
    # Display label and percentage
    text = f"{label}: {percent:.2f}%"
    cv2.putText(display_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    # Show frame
    cv2.imshow("Mask Detection", display_frame)
    
    # Wait key for normal speed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
