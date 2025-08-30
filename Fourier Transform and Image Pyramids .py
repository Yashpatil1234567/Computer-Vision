
# Mount Google Drive in Colab
# ==========================
from google.colab import drive
drive.mount('/content/drive')

# Compute and Visualize the 2D Fourier Transform of an Image
# ==========================
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Read the image in grayscale (single channel image)
img = cv.imread('/content/drive/MyDrive/OpenCV/virat .jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "Image not found. Check path carefully."

# Perform 2D Fast Fourier Transform (FFT) - converts image from spatial to frequency domain
f = np.fft.fft2(img)

# Shift the zero-frequency (low frequency/DC component) to the center
fshift = np.fft.fftshift(f)

# Compute magnitude spectrum (use log to compress range for better visualization)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Show original image and magnitude spectrum
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


# ==========================
# Apply Frequency Domain Filtering (Low-Pass and High-Pass Filters)
# ==========================
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image in grayscale
img = cv2.imread('/content/drive/MyDrive/OpenCV/virat .jpg', 0)

# Perform Discrete Fourier Transform (DFT)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift zero-frequency component to the center
dft_shift = np.fft.fftshift(dft)

# Get image size and center
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# ---------------- LOW PASS FILTER (LPF) ----------------
# Create a mask with a filled white circle in the center (allowing low frequencies)
mask_lpf = np.zeros((rows, cols, 2), np.uint8)
radius_lpf = 30  # cutoff frequency (try changing this value)
cv2.circle(mask_lpf, (ccol, crow), radius_lpf, (1, 1), -1)

# ---------------- HIGH PASS FILTER (HPF) ----------------
# Create a mask with everything white and black circle in center (blocking low frequencies)
mask_hpf = np.ones((rows, cols, 2), np.uint8)
radius_hpf = 30
cv2.circle(mask_hpf, (ccol, crow), radius_hpf, (0, 0), -1)

# Apply masks
fshift_lpf = dft_shift * mask_lpf
fshift_hpf = dft_shift * mask_hpf

# Inverse DFT (to convert back to image)
# LPF result
f_ishift_lpf = np.fft.ifftshift(fshift_lpf)
img_back_lpf = cv2.idft(f_ishift_lpf)
img_back_lpf = cv2.magnitude(img_back_lpf[:, :, 0], img_back_lpf[:, :, 1])

# HPF result
f_ishift_hpf = np.fft.ifftshift(fshift_hpf)
img_back_hpf = cv2.idft(f_ishift_hpf)
img_back_hpf = cv2.magnitude(img_back_hpf[:, :, 0], img_back_hpf[:, :, 1])

# Display results
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_back_lpf, cmap='gray')
plt.title('Low-Pass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back_hpf, cmap='gray')
plt.title('High-Pass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()


# ==========================
# Gaussian and Laplacian Pyramids of an Image
# ==========================
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Read original image (color image)
img = cv2.imread('/content/drive/MyDrive/OpenCV/virat .jpg')

# --------- Gaussian Pyramid ---------
gaussian_pyramid = [img]  # first level = original image
for i in range(5):  # create 5 levels
    img = cv2.pyrDown(img)  # downsample (reduce size)
    gaussian_pyramid.append(img)

# --------- Laplacian Pyramid ---------
# Start with the last (smallest) Gaussian level
laplacian_pyramid = [gaussian_pyramid[-1]]

# Build Laplacian by subtracting expanded Gaussian levels
for i in range(len(gaussian_pyramid) - 1, 0, -1):
    gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])  # upsample (expand size)
    
    # Make sure sizes match (sometimes pyrUp may not exactly match original size)
    if gaussian_expanded.shape != gaussian_pyramid[i - 1].shape:
        gaussian_expanded = cv2.resize(gaussian_expanded, 
                                       (gaussian_pyramid[i - 1].shape[1], 
                                        gaussian_pyramid[i - 1].shape[0]))
    
    # Laplacian = current Gaussian - expanded next Gaussian
    laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
    laplacian_pyramid.append(laplacian)

# --------- Display Gaussian Pyramid ---------
for i, level in enumerate(gaussian_pyramid):
    cv2_imshow(level)

# --------- Display Laplacian Pyramid ---------
for i, level in enumerate(laplacian_pyramid):
    display_laplacian = cv2.convertScaleAbs(level)  # adjust contrast for visibility
    cv2_imshow(display_laplacian)

# --------- Reconstruct the Image from Laplacian Pyramid ---------
img_reconstructed = laplacian_pyramid[0]  # start with smallest image
for i in range(1, len(laplacian_pyramid)):
    expanded = cv2.pyrUp(img_reconstructed)  # expand current level
    
    # Adjust size if needed
    if expanded.shape != laplacian_pyramid[i].shape:
        expanded = cv2.resize(expanded, 
                              (laplacian_pyramid[i].shape[1], 
                               laplacian_pyramid[i].shape[0]))
    
    # Add with Laplacian to reconstruct
    img_reconstructed = cv2.add(expanded, laplacian_pyramid[i])

# Show reconstructed image
cv2_imshow(img_reconstructed)
