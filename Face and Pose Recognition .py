# --- Install required libraries (only needed in Google Colab) ---
# !pip install mediapipe opencv-python

# --- Import required libraries ---
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Path to your Google Drive folder containing images ---
# Change this path if you want to test with your own images
drive_image_folder = '/content/drive/MyDrive/OpenCV'

# --- Get all image files (PNG, JPG, JPEG) from the folder ---
image_files = [
    os.path.join(drive_image_folder, f)
    for f in os.listdir(drive_image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

print(f"Found {len(image_files)} images in your Drive folder.")

# --- Initialize MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Detect faces in static images (not webcam)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# --- Function to estimate head pose using landmarks ---
def estimate_head_pose(landmarks, image_shape):
    """
    Estimate the head pose using selected face landmarks.
    """
    # Choose 6 important points from the face
    image_points = np.array([
        landmarks[1],   # Nose tip
        landmarks[33],  # Left eye corner
        landmarks[263], # Right eye corner
        landmarks[61],  # Left mouth corner
        landmarks[291], # Right mouth corner
        landmarks[0]    # Chin
    ], dtype='double')

    # 3D model points (approximate human face measurements)
    model_points = np.array([
        (0.0, 0.0, 0.0),       # Nose tip
        (-30.0, -65.5, -5.0),  # Left eye
        (30.0, -65.5, -5.0),   # Right eye
        (-40.0, -105.0, -5.0), # Left mouth
        (40.0, -105.0, -5.0),  # Right mouth
        (0.0, -133.0, -5.0)    # Chin
    ])

    # Camera matrix (assumes no distortion)
    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype='double')

    dist_coeffs = np.zeros((4, 1))  # No distortion

    # SolvePnP calculates head pose (rotation + translation)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    return rotation_vector, translation_vector, camera_matrix, dist_coeffs

# --- Process each image ---
for file_path in image_files:
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error loading image: {file_path}")
        continue

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks_2d = []
            for lm in face_landmarks.landmark:
                # Convert normalized coords to pixel coords
                x, y = int(lm.x * image_rgb.shape[1]), int(lm.y * image_rgb.shape[0])
                landmarks_2d.append((x, y))

            # Make sure enough points are detected
            if len(landmarks_2d) < 292:
                print(f"Not enough landmarks detected in {os.path.basename(file_path)}")
                continue

            # Estimate head pose
            r_vec, t_vec, cam_mtx, dist = estimate_head_pose(landmarks_2d, image_rgb.shape)

            # Copy image for annotation
            annotated_image = image_rgb.copy()

            # Draw face mesh
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Define 3D axis (X, Y, Z)
            axis_points_3d = np.float32([
                [100, 0, 0],  # X-axis
                [0, 100, 0],  # Y-axis
                [0, 0, 100]   # Z-axis
            ]).reshape(-1, 3)

            # Project 3D points into 2D image
            imgpts, _ = cv2.projectPoints(axis_points_3d, r_vec, t_vec, cam_mtx, dist)

            # Nose tip as origin
            nose_tip = landmarks_2d[1]
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Draw axis lines
            cv2.line(annotated_image, nose_tip, tuple(imgpts[0]), (0, 0, 255), 3)   # X-axis (Red)
            cv2.line(annotated_image, nose_tip, tuple(imgpts[1]), (0, 255, 0), 3)   # Y-axis (Green)
            cv2.line(annotated_image, nose_tip, tuple(imgpts[2]), (255, 0, 0), 3)   # Z-axis (Blue)

            # Show result
            plt.figure(figsize=(10, 7))
            plt.imshow(annotated_image)
            plt.title(f"Face and Pose Estimation: {os.path.basename(file_path)}")
            plt.axis('off')
            plt.show()
    else:
        print(f"No face detected in {os.path.basename(file_path)}")
