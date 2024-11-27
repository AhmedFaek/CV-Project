import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

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


def adjustBrightness(img):
    avg_brightness = np.mean(img)
    print(f"average brightness: {avg_brightness}")

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


def barcodeErosion(img):
    structuring_element_length = img.shape[0]
    vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, structuring_element_length))
    img = cv.erode(img, vertical_structure, iterations=1)

    return img


def detectingBarCode(img):
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

    # # Draw the contour and bounding box on the original image
    # output = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # cv.drawContours(output, [barcode_contour], -1, (0, 255, 0), 2)

    # # Display the results
    # images = [img, imgBlur, edges, closed, output, cropped_barcode]
    # titles = [
    #     "Original Image",
    #     "Blurred Image",
    #     "Edges",
    #     "Morphed Image",
    #     "Detected Barcode",
    #     "Final Output"
    # ]

    # plt.figure(figsize=(15, 10))
    # for i in range(len(images)):
    #     plt.subplot(2, 3, i + 1)
    #     if i == len(images) - 1:  # Last image is cropped barcode
    #         plt.imshow(images[i], cmap="gray")
    #     else:
    #         plt.imshow(images[i], cmap="gray")
    #     plt.title(titles[i])
    #     plt.axis("off")

    # plt.tight_layout()
    # plt.show()
    return cropped_barcode


def objectDetection(img,flag):
    flattened = img.flatten()

    # # Count the frequency of each pixel intensity value
    # unique, counts = np.unique(flattened, return_counts=True)

    # # Create a dictionary mapping each intensity value to its frequency
    # freq_dict = dict(zip(unique, counts))

    # Create a mask for pixels in the specified range
    lower_bound = 100
    upper_bound = 200
    mask = (img >= lower_bound) & (img <= upper_bound)

    # Use the mask to sum pixel values in the range
    total_sum = np.sum(img[mask])
    print(total_sum) 

    if total_sum >= 2000000 and total_sum<=3000000:
        flag = 1
    
    return img,flag



img = cv.imread("03 - eda ya3am ew3a soba3ak mathazarsh.jpg",0)
# 01 - lol easy.jpg
# 02 - still easy.jpg
# 03 - eda ya3am ew3a soba3ak mathazarsh.jpg
# 04 - fen el nadara.jpg
# 05 - meen taffa el nour!!!.jpg
# 06 - meen fata7 el nour 333eenaaayy.jpg
# 07 - mal7 w felfel.jpg
# 08 - compresso espresso.jpg
# 09 - e3del el soora ya3ammm.jpg
# 10 - wen el kontraastttt.jpg
# 11 - bayza 5ales di bsara7a.jpg

img = adjustBrightness(img)
img = noiseDetection(img)
img, flag = rotationDetection(img, 0)
print(flag)
img = sharpen_if_needed(img)
img, flag = objectDetection(img,flag)
print(flag)
if(flag):
    ret, img = cv.threshold(img, 25, 255, cv.THRESH_BINARY)
    img = detectingBarCode(img)
    img = barcodeErosion(img)
else:
    img = detectingBarCode(img)

plt.imshow(img,cmap='gray')
plt.show()


