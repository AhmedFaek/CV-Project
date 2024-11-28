from functions import *

img = cv.imread("09 - e3del el soora ya3ammm.jpg", 0)

img = adjustBrightness(img)

img, flag_2 = contrastDetection(img)

if flag_2 == 2:
    blurredImg = blur(img)
    cropped_barcode = detectingBarCode(blurredImg, flag_2)
    decoded_value = decode_barcode(cropped_barcode)
    print(f"Decoded Barcode: {decoded_value}")
    exit()

img = noiseDetection(img)
img, flag = rotationDetection(img, 0)
img = decompress(img)
img = sharpen_if_needed(img)
img, flag = objectDetection(img, flag)

if flag == 1:
    cropped_barcode = detectingBarCode(img, 0)
    img = barcodeErosion(cropped_barcode)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 200))  # Adjust (width, height)
    closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    dilated = cv.dilate(closed, kernel, iterations=2)
    plt.title("Final Output")
    plt.imshow(dilated, "gray")
    plt.show()

    # Decode the barcode
    decoded_value = decode_barcode(dilated)
    print(f"Decoded Barcode: {decoded_value}")

elif flag == 0:
    print(flag)
    cropped_barcode = detectingBarCode(img, 0)
    ret, img = cv.threshold(cropped_barcode, 110, 255, cv.THRESH_BINARY)
    img = img[0:195]

    plt.imshow(img, "gray")
    plt.show()

    # Decode the barcode
    decoded_value = decode_barcode(img)
    print(f"Decoded Barcode: {decoded_value}")
