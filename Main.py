from functions import *

img = cv.imread("03 - eda ya3am ew3a soba3ak mathazarsh.jpg", 0)

img = adjustBrightness(img)
img, flag_2 = contrastDetection(img)
if flag_2 == 2:
    blurredImg = blur(img)
    detectingBarCode(blurredImg, flag_2)
    exit()
img = noiseDetection(img)
img, flag = rotationDetection(img, 0)
img = sharpen_if_needed(img)
img, flag = objectDetection(img, flag)
if flag == 1:
    ret, img = cv.threshold(img, 25, 255, cv.THRESH_BINARY)
    img = detectingBarCode(img, 0)
    img = barcodeErosion(img)
    plt.title("Final Output")
    plt.imshow(img, "gray")
    plt.show()
elif flag == 0:
    img = detectingBarCode(img, 0)
