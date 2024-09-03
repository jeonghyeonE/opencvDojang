import cv2

img = cv2.imread('opencvDojang/data2/opencv-logo-white.png', cv2.IMREAD_UNCHANGED)
src = img[:,:,:3]
mask = img[:,:,3]
dst = cv2.imread('opencvDojang/data2/cat.bmp', cv2.IMREAD_COLOR)

x, y = 0, 0
h, w = src.shape[:2]
roi = dst[y:y+h, x:x+w]

cv2.copyTo(src, mask, roi)

cv2.imshow('imh',img)
cv2.imshow('mask',mask)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()