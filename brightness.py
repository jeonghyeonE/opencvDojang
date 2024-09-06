import cv2
import numpy as np

src = cv2.imread('opencvDojang/data2/cat.png', cv2.IMREAD_GRAYSCALE)
dst1 = cv2.add(src, 100)
dst2 = np.clip(src.astype(np.int16) + 100, 0, 255).astype(np.uint8)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()