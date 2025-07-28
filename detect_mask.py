import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load trained model
try:
    model = load_model('mask_detector_mobilenetv2.keras')
except Exception as e:
    print("[ERROR] Model load failed:", e)
    exit()

# Get model input size
input_size = model.input_shape[1:3]

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Starting webcam. Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        try:
            resized = cv2.resize(face_img, input_size)
        except Exception as e:
            print("Resize error:", e)
            continue

        img = resized / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)[0][0]

        if pred < 0.5:
            label = f"Mask ({(1 - pred) * 100:.1f}%)"
            color = (0, 255, 0)
        else:
            label = f"No Mask ({pred * 100:.1f}%)"
            color = (0, 0, 255)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Border text (black)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)

        # Actual label text (color)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Confidence progress bar below face
        bar_width = int(w * (1 - pred) if pred < 0.5 else w * pred)
        cv2.rectangle(frame, (x, y + h + 10),
                      (x + bar_width, y + h + 20), color, -1)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('😷 Face Mask Detector - Press Q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
