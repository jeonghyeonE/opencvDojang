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

# 트랙바 콜백 함수 (아무 기능도 없지만 필수)
def nothing(x):
    pass

# 윈도우 생성
cv2.namedWindow('Chromakey Settings')

# 트랙바 생성 (HSV 값 조정용)
cv2.createTrackbar('Lower H', 'Chromakey Settings', 35, 180, nothing)
cv2.createTrackbar('Lower S', 'Chromakey Settings', 100, 255, nothing)
cv2.createTrackbar('Lower V', 'Chromakey Settings', 100, 255, nothing)
cv2.createTrackbar('Upper H', 'Chromakey Settings', 85, 180, nothing)
cv2.createTrackbar('Upper S', 'Chromakey Settings', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Chromakey Settings', 255, 255, nothing)

# 크로마키 효과 상태 변수
chromakey_enabled = True

while True:
    
    # 첫 번째 비디오 프레임 읽기 (배경 비디오)
    ret1, frame1 = cap1.read()
    if not ret1:
        break

    # 두 번째 비디오 프레임 읽기 (크로마키 비디오)
    ret2, frame2 = cap2.read()
    if not ret2:
        break

    # 프레임 크기 조정
    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))

    # 트랙바에서 HSV 값 읽기
    lower_h = cv2.getTrackbarPos('Lower H', 'Chromakey Settings')
    lower_s = cv2.getTrackbarPos('Lower S', 'Chromakey Settings')
    lower_v = cv2.getTrackbarPos('Lower V', 'Chromakey Settings')
    upper_h = cv2.getTrackbarPos('Upper H', 'Chromakey Settings')
    upper_s = cv2.getTrackbarPos('Upper S', 'Chromakey Settings')
    upper_v = cv2.getTrackbarPos('Upper V', 'Chromakey Settings')

    # HSV에서 초록색에 해당하는 색상 범위 설정
    lower_green = np.array([lower_h, lower_s, lower_v])
    upper_green = np.array([upper_h, upper_s, upper_v])

    if chromakey_enabled:
        # 두 번째 영상을 HSV로 변환
        hsv_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        # 초록색 배경에 대한 마스크 생성 (HSV 색상 범위로 추출)
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # 마스크 반전 (초록색을 제외한 부분)
        mask_inv = cv2.bitwise_not(mask)

        # 초록색 배경을 제거한 전경 추출
        frame2_fg = cv2.bitwise_and(frame2, frame2, mask=mask_inv)

        # 배경 비디오에서 초록색 부분을 채우기 위한 배경 추출
        frame1_bg = cv2.bitwise_and(frame1, frame1, mask=mask)

        # 배경과 전경을 합성
        result = cv2.add(frame1_bg, frame2_fg)
    else:
        # 크로마키 효과가 꺼져 있으면 두 번째 영상 그대로 출력
        result = frame2

    # 결과 영상 출력
    cv2.imshow('Chromakey Video with HSV', result)

    # 'c' 키를 눌러 크로마키 효과 On/Off
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
