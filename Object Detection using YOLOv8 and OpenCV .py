# Mount Google Drive to access dataset/images
from google.colab import drive
drive.mount('/content/drive')

# Install the ultralytics library (YOLOv8)
!pip install ultralytics

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob

# Load pre-trained YOLOv8 model (nano version - small and fast)
model = YOLO('yolov8n.pt')

# Function to run YOLO on images
def run_yolo_on_images(image_paths):
    for img_path in image_paths:
        # Run detection
        results = model(img_path)

        # Display results using YOLO's built-in plotting
        for r in results:
             im_array = r.plot()  # predictions with bounding boxes
             im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
             plt.figure(figsize=(10,7))
             plt.imshow(im_rgb)
             plt.title(f"Detections: {img_path}")
             plt.axis('off')
             plt.show()

# Load images from Google Drive folder (change path as needed)
custom_images = glob.glob('/content/drive/MyDrive/OpenCV/*109.jpg')

# Run YOLO on selected images
run_yolo_on_images(custom_images)
