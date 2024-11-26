import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

flag = 0

def checkRGB(image):
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Calculate the absolute difference between the R, G, and B channels
    diff1 = np.abs(image_rgb[:, :, 0] - image_rgb[:, :, 1])  # R - G
    diff2 = np.abs(image_rgb[:, :, 0] - image_rgb[:, :, 2])  # R - B
    diff3 = np.abs(image_rgb[:, :, 1] - image_rgb[:, :, 2])  # G - B

    # Any non-zero difference means a pixel is not pure grayscale (not black or white)
    has_color = np.any((diff1 > 0) | (diff2 > 0) | (diff3 > 0))

    if has_color:
        return True
    else:
        return False

def removeOverlay(img):
    _, img = cv.threshold(img, 15, 255, cv.THRESH_BINARY)
    flag = 1


    return img, flag

def barcodeErosion(img):
    # Erode the barcode image
    structuring_element_length = img.shape[0]
    vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, structuring_element_length))
    dilated_image = cv.erode(img, vertical_structure, iterations=1)

    # Apply threshold
    ret, dilated_image = cv.threshold(dilated_image, 10, 150, cv.THRESH_BINARY)

    return dilated_image

def rotationDetection(img, flag):
    padded_img = cv.copyMakeBorder(
        img,
        top=40,
        bottom=40,
        left=40,
        right=40,
        borderType=cv.BORDER_CONSTANT,
        value=[248, 248, 248]
    )

    # Blur and detect edges
    imgBlur = cv.GaussianBlur(padded_img, (5, 5), 0)
    edges = cv.Canny(imgBlur, 30, 100)

    # Use Hough Line Transform to detect the lines
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

    # Calculate the angle of the detected lines and compute the average angle
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 180  # Convert from radians to degrees
        angles.append(angle)

    # Find the median angle (to avoid outliers)
    angle = np.median(angles)

    if abs(angle) != 180:
        # Rotate the image to straighten the barcode
        (h, w) = padded_img.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation and fill the image with white color
        rotated_img = cv.warpAffine(padded_img, rotation_matrix, (w, h), flags=cv.INTER_CUBIC,
                                    borderMode=cv.BORDER_CONSTANT, borderValue=(248, 248, 248))
        flag = 1

        return rotated_img, flag

    else:
        return img, flag

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
    flag = 0

    if checkRGB(img):
        img, flag = removeOverlay(img)



    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    img = adjustBrightness(img)

    img = noiseDetection(img)

    img, flag = rotationDetection(img, flag)





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
    y = max(0, y - padding - 5)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    # Crop the barcode without numbers (exclude bottom portion)
    cropped_barcode = img[y:y + int(h), x:x + int(w)]

    # Draw the contour and bounding box on the original image
    output = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.drawContours(output, [barcode_contour], -1, (0, 255, 0), 2)

    output = sharpen_if_needed(output)


    if flag:
        cropped_barcode = barcodeErosion(cropped_barcode)



    # Display the results
    images = [img, imgBlur, edges, closed, output, cropped_barcode]
    titles = [
        "Original Image",
        "Blurred Image",
        "Edges",
        "Morphed Image",
        "Detected Barcode",
        "Final Output"
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



