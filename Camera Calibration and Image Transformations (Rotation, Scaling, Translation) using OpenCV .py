
# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Import required libraries ---
import cv2
import numpy as np
import glob
from google.colab.patches import cv2_imshow

# 1. CAMERA CALIBRATION (Find Intrinsic Parameters)
# ======================================================

# Prepare a grid of 3D points (chessboard corners in real world space)
# Example points: (0,0,0), (1,0,0), (2,0,0), ... (6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store 3D points (real world) and 2D points (image plane)
objpoints = []
imgpoints = []

# Load all chessboard images from folder
images = glob.glob('/content/drive/MyDrive/OpenCV/chess_board/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect chessboard corners (pattern 7x6)
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw corners and display
        cv2.drawChessboardCorners(img, (7,6), corners, ret)
        cv2_imshow(img)

# Perform calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("=== CAMERA CALIBRATION RESULTS ===")
print("Camera matrix:\n", mtx)   # Intrinsic parameters
print("Distortion coefficients:\n", dist)

# 2. CAMERA CALIBRATION (With corner refinement)
# ======================================================

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    if ret:
        objpoints.append(objp)

        # Refine corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and show refined corners
        cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2_imshow(img)


# 3. IMAGE ROTATION
# ======================================================

if images:
    image_path = images[0]   # Use first image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Rotate 90 degrees clockwise
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Rotate 180 degrees
    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)

    # Rotate by arbitrary angle (e.g., 45 degrees)
    center = (width // 2, height // 2)
    angle = 45
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_arbitrary = cv2.warpAffine(image, rotation_matrix, (width, height))

    print("\n=== IMAGE ROTATION ===")
    cv2_imshow(image)
    cv2_imshow(rotated_90)
    cv2_imshow(rotated_180)
    cv2_imshow(rotated_arbitrary)
else:
    print("No images found for rotation demo.")


# 4. IMAGE SCALING
# ======================================================

# Resize (scale) image
scaled_half = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
scaled_double = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
scaled_specific = cv2.resize(image, (300, 200), interpolation=cv2.INTER_LINEAR)

print("\n=== IMAGE SCALING ===")
cv2_imshow(image)
cv2_imshow(scaled_half)
cv2_imshow(scaled_double)
cv2_imshow(scaled_specific)

# 5. IMAGE TRANSLATION (Shifting)
# ======================================================

tx = 100  # Shift right
ty = 50   # Shift down
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

print("\n=== IMAGE TRANSLATION ===")
cv2_imshow(translated_image)
