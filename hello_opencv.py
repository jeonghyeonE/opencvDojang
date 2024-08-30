import sys
import cv2

# opencv 버전 확인
print('Hello OpenCV', cv2.__version__)

# image read
img_gray = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.imread('./data/lenna.bmp')

if img_gray is None or img_bgr is None:
    print('Image load failed!')
    sys.exit()

cv2.namedWindow('img_gray')
cv2.namedWindow('img_bgr')
cv2.imshow('img_gray', img_gray)
cv2.imshow('img_bgr', img_bgr)
cv2.waitKey()
cv2.destroyAllWindows()