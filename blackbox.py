import cv2
import os
import time

# 원본 동영상 파일 경로
video_file = 'opencvDojang/data/video.mp4'  # 여기에 사용할 동영상 파일 이름을 적어주세요

# 출력 동영상 저장 폴더
output_folder = 'opencvDojang/output_videos'
os.makedirs(output_folder, exist_ok=True)

# 동영상 파일을 읽기 위해 VideoCapture 객체를 생성
cap = cv2.VideoCapture(video_file)

# 동영상의 프레임 레이트(fps) 및 해상도 정보를 가져옴
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 파일명에 사용할 기본 문자열
base_filename = '20240903_'

# 비디오 파일을 분할
part_number = 0
start_time = time.time()
out = None

while True:
    # 현재 프레임을 읽음
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 현재 시간과 시작 시간의 차이를 계산
    elapsed_time = time.time() - start_time

    # 60초가 지났거나 처음 파일을 생성하는 경우
    if elapsed_time >= 60 or out is None:
        if out is not None:  # 이전 파일이 있을 경우 닫기
            out.release()

        # 새로운 파일명 생성
        output_filename = f'{base_filename}{part_number}.avi'
        output_path = os.path.join(output_folder, output_filename)

        # 새로운 VideoWriter 객체를 생성하여 동영상을 저장
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        part_number += 1
        start_time = time.time()  # 새로운 파일의 시작 시간을 기록

    # 현재 프레임을 새로운 파일에 기록
    out.write(frame)

# 마지막 파일을 닫음
if out is not None:
    out.release()

cap.release()
cv2.destroyAllWindows()

print(f"총 {part_number}개의 비디오 파일이 생성되었습니다.")