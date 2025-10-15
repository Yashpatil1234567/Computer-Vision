import cv2
import matplotlib.pyplot as plt

# Step 1: Load image (update path as needed)
image_path = '/content/drive/MyDrive/OpenCV/download.jpg'  # Put your image path here
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not loaded. Check the file path.")
else:
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Step 4: Threshold to get binary image (objects will be white)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 5: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Draw contours and count objects
    output = image.copy()
    for i, contour in enumerate(contours):
        # Draw contour
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        # Calculate center for numbering
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(output, str(i+1), (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

    # Step 7: Print the number of objects detected
    print(f"🔢 Number of objects detected: {len(contours)}")

    # Step 8: Show original and output images side-by-side
    plt.figure(figsize=(12,6))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Objects Counted')
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
