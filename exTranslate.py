import cv2, sys
import numpy as np

img = cv2.imread('opencvDojang/data2/lenna.bmp')

if img is None:
    sys.exit('Image load failed')
    
aff = np.array([[1,0,200],[0,1,100]], dtype=np.float32)
dst = cv2.warpAffine(img,aff,(0,0))

cv2.imshow('img', dst)
cv2.waitKey()
cv2.destroyAllWindows()
