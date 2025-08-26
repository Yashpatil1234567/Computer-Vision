from google.colab import drive
drive.mount('/content/drive')            

import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Load two images to compare
# img1 = Source image, img2 = Destination image
img1 = cv2.imread('/content/drive/MyDrive/OpenCV/TAJ.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/content/drive/MyDrive/OpenCV/TAJ2.jpg', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded correctly
if img1 is None or img2 is None:
    print("Error: Unable to load images")
    exit()

# Step 1: Detect Keypoints using Harris Corner Detection
# Harris detects corners as keypoints (requires grayscale image)
harris_corners = cv2.cornerHarris(img1, blockSize=2, ksize=3, k=0.04)
harris_corners = cv2.dilate(harris_corners, None)  # Dilate to make corners more visible

# Create a copy of the original image to draw Harris corners
img1_harris = img1.copy()
img1_harris[harris_corners > 0.01 * harris_corners.max()] = 255  # Mark strong corners in white

# Step 2: Detect Keypoints using SIFT (Scale-Invariant Feature Transform)
sift = cv2.SIFT_create()
keypoints1_sift, descriptors1_sift = sift.detectAndCompute(img1, None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(img2, None)

# Step 3: Match Features
# Use Brute-Force matcher for SIFT descriptors (NORM_L2 is used for SIFT/float descriptors)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors between the two images
matches_sift = bf.match(descriptors1_sift, descriptors2_sift)

# Sort matches by distance (best matches first)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)

# Step 4: Visualize Matches
# Draw top 50 matches between img1 and img2
img_matches_sift = cv2.drawMatches(
    img1, keypoints1_sift,
    img2, keypoints2_sift,
    matches_sift[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Step 5: Display Results
plt.figure(figsize=(10, 10))

# Show Harris Corners
plt.subplot(2, 2, 1)
plt.imshow(img1_harris, cmap='gray')
plt.title('Harris Corners')

# Show SIFT Matches
plt.subplot(2, 2, 2)
plt.imshow(img_matches_sift)
plt.title('SIFT Matches')

plt.tight_layout()
plt.show()
