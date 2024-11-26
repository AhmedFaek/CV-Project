
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def noiseDetection(img):
    edges = cv.Canny(img ,30 ,100)
    num_edges = np.count_nonzero(edges)
    print(num_edges)

    if num_edges > 50000:
        blurred = cv.blur(img, (1, 7))

        # Apply cv.medianBlur
        filtered = cv.medianBlur(blurred, 3)  # ksize must be odd

        # Apply thresholding
        retval, th = cv.threshold(filtered, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        erosion = cv.dilate(th, (1, 5), iterations=5)

        return erosion
    else:
        return img

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

def adjustBrightness(img):
    avg_brightness = np.mean(img)
    print(f"avg brightness {avg_brightness}")

    # Check if the image has low brightness
    if avg_brightness < 20:
        # Create a binary mask where the pixels greater than 0 are set to 1
        mask = img > 10

        # Create a new image initialized to the same values as the original image
        bright_image = img.copy()

        # Set all pixels where mask is True to 255
        bright_image[mask] = 255

        return bright_image

    elif avg_brightness > 250:
        mask = img >= 250

        # Create a new image initialized to the same values as the original image
        dark_image = img.copy()

        # Set all pixels where mask is True to 255
        dark_image[mask] = 255

        # Set all pixels where mask is False to 0 (optional, since the default is already 0)
        dark_image[~mask] = 0
        return dark_image

    else:
        print(f"no need to adjust brightness")
        return img


def detectingBarCode(img):
    img = noiseDetection(img)
    img = adjustBrightness(img)
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
        return None

    # Get the minimum area rectangle
    rect = cv.minAreaRect(barcode_contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    # Calculate the rotation angle
    angle = rect[-1]
    if rect[1][0] < rect[1][1]:  # Width < Height
        angle = -(90 + angle)
    else:  # Width > Height
        angle = -angle

    print(f"Rotation Angle: {angle} degrees")

    # Rotate the entire image to make the barcode parallel to the X-axis
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    # Re-detect contours on the rotated image for accurate cropping
    imgBlur = cv.GaussianBlur(rotated, (5, 5), 0)
    edges = cv.Canny(imgBlur, 30, 100)
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    if len(contours) > 0:
        barcode_contour = contours[0]

    # Get the bounding box for the barcode
    x, y, w, h = cv.boundingRect(barcode_contour)

    # Optionally expand the bounding box to include full barcode
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(rotated.shape[1] - x, w + 2 * padding)
    h = min(rotated.shape[0] - y, h + 2 * padding)

    # Crop the barcode
    cropped_barcode = rotated[y:y + h, x:x + w]

    # Visualize the result
    images = [rotated, imgBlur, edges, closed, cropped_barcode]
    titles = [
        "Rotated Image",
        "Blurred Image",
        "Edges",
        "Morphed Image",
        "Cropped Barcode"
    ]

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    
