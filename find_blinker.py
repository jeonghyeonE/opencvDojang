import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread("opencvDojang/data2/all.jpg")

# BGR 이미지를 HSV로 변환
hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 신호등의 빨간색, 노란색, 녹색 범위 정의 (HSV 값)
# 빨간색 범위 (2개의 범위: 저채도와 고채도)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# 노란색 범위
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# 녹색 범위
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# 빨간색 마스크
mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # 두 범위 합치기

# 노란색 마스크
mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

# 녹색 마스크
mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)

# 마스크에서 각 색상의 픽셀 수 계산
red_count = cv2.countNonZero(mask_red)
yellow_count = cv2.countNonZero(mask_yellow)
green_count = cv2.countNonZero(mask_green)

# 가장 많이 탐지된 색상에 따라 신호등 색상 출력
if red_count > yellow_count and red_count > green_count:
    print("빨간 신호 탐지됨")
    detected_color = "Red"
elif yellow_count > red_count and yellow_count > green_count:
    print("노란 신호 탐지됨")
    detected_color = "Yellow"
elif green_count > red_count and green_count > yellow_count:
    print("녹색 신호 탐지됨")
    detected_color = "Green"
else:
    print("신호등 색상이 감지되지 않음")
    detected_color = "None"

# 결과 출력용 이미지에 마스크 적용하여 표시
result_red = cv2.bitwise_and(img, img, mask=mask_red)
result_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
result_green = cv2.bitwise_and(img, img, mask=mask_green)

# 원본 이미지와 결과 이미지 출력
cv2.imshow('Original Image', img)
cv2.imshow('Red Detection', result_red)
cv2.imshow('Yellow Detection', result_yellow)
cv2.imshow('Green Detection', result_green)

# ESC 키를 누르면 종료
cv2.waitKey(0)
cv2.destroyAllWindows()
