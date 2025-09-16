
# Mount Google Drive (needed only if you are using Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Optical Flow and Motion Estimation using OpenCV

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

# ==== Upload Video ====
# This lets you upload a video file from your computer into Colab
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# === Open video capture ===
cap = cv2.VideoCapture(video_path)

# If you want to use webcam instead of uploading a video, uncomment below line
# cap = cv2.VideoCapture(0)

# === Parameters for Lucas-Kanade Optical Flow ===
# These are algorithm settings for tracking motion
lk_params = dict(winSize  = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# === Parameters for Shi-Tomasi corner detection ===
# This helps to find important points in the image to track
feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7)

# === Take first frame and find corners in it ===
ret, old_frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# === Create a mask image for drawing motion lines ===
mask = np.zeros_like(old_frame)

frame_count = 0
max_frames = 100  # Limit number of frames to process

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === Calculate Optical Flow (track motion) ===
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # === Select good points ===
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # === Draw motion vectors ===
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)
    else:
        img = frame

    # === Show the frame in Colab ===
    cv2_imshow(img)
    key = cv2.waitKey(50) & 0xFF
    if key == 27:  # ESC to quit
        break

    # === Update previous frame and points ===
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Done.")
