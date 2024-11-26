#1-2-4

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def sharpen_if_needed(img):
    # Compute the Laplacian
    laplacian = cv.Laplacian(img, cv.CV_64F)

    # Compute the variance of the Laplacian
    variance = laplacian.var()
    print(f"Variance of Laplacian: {variance}")

    # Decide if the image is blurry (threshold can be adjusted)
    threshold = 300
    if variance < threshold:
        # Sharpen the image if it is blurry
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv.filter2D(img, -1, kernel)

    return img


def detectingBarCode(img):
    # Apply sharpening if needed
    img = sharpen_if_needed(img)

    imgBlur = cv.GaussianBlur(img, (5, 5), 0)

    # Adjust thresholds for better edge detection
    edges = cv.Canny(imgBlur, 30, 100)

    # Increase kernel size for better morphological closing
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    if len(contours) > 0:
        barcode_contour = contours[0]
    else:
        print("No contours detected")
        exit()

    # Get the bounding box for the largest contour
    x, y, w, h = cv.boundingRect(barcode_contour)

    # Optionally expand the bounding box to include the full barcode
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    # Crop the barcode without numbers (exclude bottom portion)
    cropped_barcode = img[y:y + int(h), x:x + int(w)]

    # Draw the contour and bounding box on the original image
    output = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.drawContours(output, [barcode_contour], -1, (0, 255, 0), 2)

    output = sharpen_if_needed(output)

    # Display the results
    images = [img, imgBlur, edges, closed, output, cropped_barcode]
    titles = [
        "Original Image",
        "Blurred Image",
        "Edges",
        "Morphed Image",
        "Detected Barcode",
        "Cropped Barcode Without Numbers"
    ]

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        if i == len(images) - 1:  # Last image is cropped barcode
            plt.imshow(images[i], cmap="gray")
        else:
            plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Read and preprocess the image
img = cv.imread("04 - fen el nadara.jpg", 0)

# Call detectingBarCode function directly; it will handle sharpening if needed
detectingBarCode(img)
