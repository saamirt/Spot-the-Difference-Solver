from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import numpy as np
import wget

FILENAME = 'difference1.jpg'
wget.download('http://www.helpmykidlearn.ie/images/uploads/spot_the_difference_bigger.jpg', FILENAME)
imageA = cv2.imread(FILENAME)

rows,cols,_ = imageA.shape
y_offset = 0

match_loc = {'sum':0}
for x_offset in range(100,cols):
    imageA = cv2.imread(FILENAME)
    M = np.float32([[1,0,x_offset],[0,1,y_offset]])
    imageB = cv2.warpAffine(imageA,M,(cols,rows))

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    tmp_sum  = np.sum(diff[y_offset:, x_offset:])
    if (tmp_sum > match_loc['sum']):
        match_loc = {'sum': tmp_sum, 'x': x_offset, 'y': y_offset}
print(match_loc)

imageA = cv2.imread('difference1.jpg')
M = np.float32([[1,0,match_loc['x']],[0,1,match_loc['y']]])
imageB = cv2.warpAffine(imageA,M,(cols,rows))

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
# (score2, diff2) = compare_ssim(imageA, imageB, full=True, multichannel=True)
# diff2 = (diff2 * 255).astype("uint8")

thresh = cv2.threshold(diff, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)

# # show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Diff", diff)
cv2.waitKey(0)