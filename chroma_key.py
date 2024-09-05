import cv2
import numpy as np

# 비디오 파일 경로 설정
video1_path = 'opencvDojang/data2/raining.mp4'  # 배경 비디오
video2_path = 'opencvDojang/data2/woman.mp4'  # 크로마키 비디오 (초록색 배경)

# 비디오 캡처 객체 생성
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# 비디오 정보 얻기
fps1 = cap1.get(cv2.CAP_PROP_FPS)
fps2 = cap2.get(cv2.CAP_PROP_FPS)
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 프레임 사이즈를 맞추기 위해 크기 조정
if width1 != width2 or height1 != height2:
    width, height = width1, height1  # 배경 영상 크기에 맞추기
else:
    width, height = width1, height1

# 크로마키 효과를 위한 색상 범위 설정 (초록색)
# lower_green = np.array([50, 100, 100])
# upper_green = np.array([70, 255, 255])

# HSV에서 초록색에 해당하는 색상 범위 설정
lower_green = np.array([35, 100, 100])  # 낮은 HSV 값
upper_green = np.array([85, 255, 255])  # 높은 HSV 값

# 크로마키 효과 상태 변수
chromakey_enabled = True

while cap1.isOpened() and cap2.isOpened():
    # ret1, frame1 = cap1.read()
    # ret2, frame2 = cap2.read()

    # if not ret1 or not ret2:
    #     break
    
    
    if chromakey_enabled:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
    
    ret2, frame2 = cap2.read()
    if not ret2:
        break

    # 프레임 크기 조정
    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))

    if chromakey_enabled:
        # HSV 색 공간으로 변환
        hsv_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        # 초록색 배경에 대한 마스크 생성
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # 마스크 반전 (배경을 제거)
        mask_inv = cv2.bitwise_not(mask)

        # 배경에서 초록색 부분을 제거
        frame2_bg_removed = cv2.bitwise_and(frame2, frame2, mask=mask_inv)

        # 배경 비디오에서 마스크로 제거한 부분 추출
        frame1_bg = cv2.bitwise_and(frame1, frame1, mask=mask)

        # 두 영상을 합성
        result = cv2.add(frame1_bg, frame2_bg_removed)
    else:
        result = frame2  # 크로마키를 끈 상태에서는 그냥 두 번째 영상 출력

    # 결과 출력
    cv2.imshow('Chromakey Video', result)

    # 'c' 키를 누르면 크로마키 효과 On/Off
    key = cv2.waitKey(30) & 0xFF
    if key == ord('c'):
        chromakey_enabled = not chromakey_enabled

    # 'q' 키를 누르면 종료
    if key == ord('q'):
        break

# 비디오 캡처 객체 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()