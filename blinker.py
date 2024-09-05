import os
import cv2
import numpy as np

img = cv2.imread("opencvDojang/data2/all.jpg")

# 트랙바 콜백 함수 (아무 기능도 없지만 필수)
def nothing(x):
    pass

# 윈도우 생성
cv2.namedWindow('Chromakey Settings', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Chromakey Settings', 600, 100)  # 창 크기를 600x400으로 설정


# 트랙바 생성 (HSV 값 조정용)
cv2.createTrackbar('Lower H', 'Chromakey Settings', 35, 180, nothing)
cv2.createTrackbar('Lower S', 'Chromakey Settings', 100, 255, nothing)
cv2.createTrackbar('Lower V', 'Chromakey Settings', 100, 255, nothing)
cv2.createTrackbar('Upper H', 'Chromakey Settings', 85, 180, nothing)
cv2.createTrackbar('Upper S', 'Chromakey Settings', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Chromakey Settings', 255, 255, nothing)

while True:
    
    # 트랙바에서 HSV 값 읽기
    lower_h = cv2.getTrackbarPos('Lower H', 'Chromakey Settings')
    lower_s = cv2.getTrackbarPos('Lower S', 'Chromakey Settings')
    lower_v = cv2.getTrackbarPos('Lower V', 'Chromakey Settings')
    upper_h = cv2.getTrackbarPos('Upper H', 'Chromakey Settings')
    upper_s = cv2.getTrackbarPos('Upper S', 'Chromakey Settings')
    upper_v = cv2.getTrackbarPos('Upper V', 'Chromakey Settings')
    
    # Lower와 Upper HSV 값으로 배열 생성
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # 이미지의 BGR을 HSV로 변환
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV 범위에 해당하는 마스크 생성
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # 원본 이미지에 마스크 적용
    result = cv2.bitwise_and(img, img, mask=mask)

    # 결과 이미지 출력
    cv2.imshow('img', mask)

    # 1ms 대기 및 키 입력 대기 (ESC 입력 시 종료)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키를 누르면 종료
        break

# 윈도우 닫기
cv2.destroyAllWindows()
