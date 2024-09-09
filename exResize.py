import cv2

ori_img = cv2.imread('opencvDojang/data2/result.png')
dst1 = cv2.resize(ori_img, (0,0), fx=0.3, fy=0.3) # 기본값 INTER_LINEAR
dst2 = cv2.resize(ori_img, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_NEAREST)
dst3 = cv2.resize(ori_img, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

# 창을 고정 크기로 만들기
cv2.namedWindow('ori_img', cv2.WINDOW_NORMAL)
cv2.namedWindow('INTER_LINEAR', cv2.WINDOW_NORMAL)
cv2.namedWindow('INTER_NEAREST', cv2.WINDOW_NORMAL)
cv2.namedWindow('INTER_AREA', cv2.WINDOW_NORMAL)

# 고정할 창 크기 설정 (예: 500x500)
cv2.resizeWindow('ori_img', 512, 512)
cv2.resizeWindow('INTER_LINEAR', 512, 512)
cv2.resizeWindow('INTER_NEAREST', 512, 512)
cv2.resizeWindow('INTER_AREA', 512, 512)

cv2.imshow('ori_img', ori_img)
cv2.imshow('INTER_LINEAR', dst1)
cv2.imshow('INTER_NEAREST', dst2)
cv2.imshow('INTER_AREA', dst3)

cv2.waitKey()
cv2.destroyAllWindows()
