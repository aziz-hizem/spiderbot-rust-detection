"""
Capture frames from the ESP32-CAM stream so you can label them and add
them to your training set later.

Use this AFTER your ESP32-CAM is up and streaming.

USAGE:
    1. Find your ESP32's IP (printed in Arduino serial monitor when it boots).
    2. Set ESP32_URL below.
    3. Run:  python capture_frames.py
    4. Walk around with the robot pointing the camera at rusty + non-rusty
       things for ~10 minutes.
    5. Frames will be saved to the 'esp32_frames/' folder.
    6. Upload them to Roboflow, label the rusty ones, retrain.
"""

import cv2
import os

ESP32_URL            = "http://192.168.1.X:81/stream"   # change to your ESP32's IP
OUTPUT_DIR           = "esp32_frames"
SAVE_EVERY_N_FRAMES  = 30   # at ~10-15 fps, this is one frame every ~2-3 seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(ESP32_URL)
if not cap.isOpened():
    print(f"ERROR: could not connect to {ESP32_URL}")
    raise SystemExit(1)

print(f"Recording from {ESP32_URL}")
print(f"Saving every {SAVE_EVERY_N_FRAMES}th frame to ./{OUTPUT_DIR}/")
print("Press ESC to stop.")

i = 0
saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    i += 1
    if i % SAVE_EVERY_N_FRAMES == 0:
        path = os.path.join(OUTPUT_DIR, f"frame_{saved:04d}.jpg")
        cv2.imwrite(path, frame)
        saved += 1
        print(f"saved {path}")

    cv2.imshow("ESP32-CAM (recording)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. {saved} frames saved to ./{OUTPUT_DIR}/")
