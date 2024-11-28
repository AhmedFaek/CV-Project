import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def display_images(images, titles, cmap="gray"):
    plt.figure(figsize=(15, 10))
    for i, img in enumerate(images):
        plt.subplot(2, (len(images) + 1) // 2, i + 1)
        if cmap and len(img.shape) == 2:  # Grayscale images
            plt.imshow(img, cmap=cmap)
        else:  # Color images
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


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
    edges = cv.Canny(img, 30, 100)
    num_edges = np.count_nonzero(edges)
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


def adjustBrightness(img):
    avg_brightness = np.mean(img)

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
        return img


def vertical_median_filter(img):
    # Create a copy of the image to store the filtered result
    filtered_image = np.zeros_like(img)

    # Loop through each column
    for col in range(img.shape[1]):
        # Get the column values
        column_values = img[:, col]

        # Calculate the median for this column
        median_value = np.median(column_values)

        # Replace the entire column with the median value
        filtered_image[:, col] = median_value

    return filtered_image


def sharpen_if_needed(img):
    # Compute the Laplacian
    laplacian = cv.Laplacian(img, cv.CV_64F)

    # Compute the variance of the Laplacian
    variance = laplacian.var()

    # Decide if the image is blurry (threshold can be adjusted)
    threshold = 300
    if variance < threshold:
        # Sharpen the image if it is blurry
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv.filter2D(img, -1, kernel)

    return img


def barcodeErosion(img):
    structuring_element_length = img.shape[0]
    vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, structuring_element_length))
    img = cv.erode(img, vertical_structure, iterations=1)

    return img


def detectingBarCode(img, flag2):
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

    padding = 0
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    # Crop the barcode without numbers (exclude bottom portion)
    cropped_barcode = img[y:y + int(h), x:x + int(w)]

    # Draw the contour and bounding box on the original image
    output = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.drawContours(output, [barcode_contour], -1, (0, 255, 0), 2)

    if flag2 == 2:
        cropped_barcode = barcodeErosion(cropped_barcode)

    # Display the results
    images = [img, imgBlur, edges, closed, output, cropped_barcode]
    titles = [
        "Original Image",
        "Blurred Image",
        "Edges",
        "Morphed Image",
        "Detected Barcode",
        "Final Output - cropped_barcode"
    ]

    display_images(images, titles)
    return cropped_barcode


def objectDetection(img, flag):
    flattened = img.flatten()

    lower_bound = 100
    upper_bound = 200
    mask = (img >= lower_bound) & (img <= upper_bound)

    # Use the mask to sum pixel values in the range
    total_sum = np.sum(img[mask])

    if 2000000 <= total_sum <= 3000000:
        flag = 1
        ret, img = cv.threshold(img, 50, 255, cv.THRESH_BINARY)

    return img, flag


def blur(img):
    imgBlur = cv.GaussianBlur(img, (5, 5), 0)
    return imgBlur


def contrastDetection(img):
    # Calculate the standard deviation of pixel intensities
    std_dev = np.std(img)

    if std_dev > 50:
        return img, 0
    else:
        min_val = np.min(img)
        max_val = np.max(img)
        stretched_image = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return stretched_image, 2


def decode_barcode(cropped_image):
    import numpy as np

    # 0 means narrow, 1 means wide
    NARROW = "0"
    WIDE = "1"

    # Map for Code 11 widths
    code11_widths = {
        "00110": "Stop/Start",
        "10001": "1",
        "01001": "2",
        "11000": "3",
        "00101": "4",
        "10100": "5",
        "01100": "6",
        "00011": "7",
        "10010": "8",
        "10000": "9",
        "00001": "0",
        "00100": "-",
    }

    print("Step 1: Convert the image to binary")
    # Convert the image to binary
    mean = cropped_image.mean(axis=0)  # Average intensity across columns
    print(f"Column-wise mean intensities: {mean}")

    mean[mean <= 127] = 1  # Black bar
    mean[mean > 128] = 0  # White bar
    print(f"Binary conversion: {mean}")

    # Convert to a string of pixels
    pixels = ''.join(mean.astype(np.uint8).astype(str))
    print(f"Binary pixel string: {pixels}")

    # Detect bar sizes
    narrow_bar_size = len(next(iter(pixels.split("0")), ""))
    wide_bar_size = narrow_bar_size * 2
    print(f"Detected narrow bar size: {narrow_bar_size}, wide bar size: {wide_bar_size}")

    digits = []
    pixel_index = 0
    current_digit_widths = ""
    skip_next = False

    print("Step 2: Start decoding")
    while pixel_index < len(pixels):
        if skip_next:
            print(f"Skipping separator bar at index {pixel_index}")
            pixel_index += narrow_bar_size
            skip_next = False
            continue

        count = 1
        while (pixel_index + count < len(pixels)) and (pixels[pixel_index] == pixels[pixel_index + count]):
            count += 1

        print(f"Bar detected: {'Black' if pixels[pixel_index] == '1' else 'White'}, width: {count}")
        current_digit_widths += NARROW if count == narrow_bar_size else WIDE
        print(f"Current digit widths: {current_digit_widths}")

        pixel_index += count

        if current_digit_widths in code11_widths:
            digit = code11_widths[current_digit_widths]
            digits.append(digit)
            print(f"Decoded digit: {digit}")
            current_digit_widths = ""
            skip_next = True  # Skip the separator bar

    print(f"Final decoded digits: {''.join(digits)}")
    return ''.join(digits)
