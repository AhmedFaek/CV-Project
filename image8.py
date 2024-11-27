import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def adjust_contrast(img, factor=1.5):
    """
    Increase the contrast by multiplying pixel values by a factor.
    A factor greater than 1 increases the contrast.
    """
    img = img.astype(np.float32)  # Convert to float for calculation
    img = img * factor  # Multiply pixel values by the factor
    img = np.clip(img, 0, 255)  # Ensure the pixel values stay within [0, 255]
    img = img.astype(np.uint8)  # Convert back to uint8
    return img


def apply_median_filter(img, kernel_size=3):
    # Apply the median filter
    filtered_img = cv.medianBlur(img, kernel_size)

    return filtered_img


# Example usage:
img = cv.imread("08 - compresso espresso.jpg", 0)

# Step 1: Histogram stretching
stretched_img = adjust_contrast(img)
plt.imshow(stretched_img, cmap='gray')
plt.title('Histogram Stretched Image')
plt.show()

# Step 2: Median filtering
filtered_img = apply_median_filter(stretched_img, kernel_size=3)
plt.imshow(filtered_img, cmap='gray')
plt.title('Median Filtered Image')
plt.show()
