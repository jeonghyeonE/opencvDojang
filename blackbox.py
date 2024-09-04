import os
import cv2
import time
import shutil
from datetime import datetime

# 원본 동영상 파일 경로
video_file = 'opencvDojang/data/video.mp4'
capture = cv2.VideoCapture(video_file)

# 녹화 동영상 폴더 경로
base_directory = 'opencvDojang/output_videos'

# 동영상 정보 가져오기
fps = capture.get(cv2.CAP_PROP_FPS)  # 프레임 속도
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이
total_frames_per_minute = int(fps * 60)  # 1분에 해당하는 프레임 수

# 녹화 관련 설정
max_folder_size = 500 * 1024 * 1024  # 500MB
record_duration = 60  # 1분 (60초)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 동영상 코덱

# 폴더 생성
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 폴더 크기 체크
def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# 500MB 초과시 폴더 삭제
def remove_oldest_directory(base_directory):
    folders = [os.path.join(base_directory, d) for d in os.listdir(base_directory)]
    if folders:
        oldest_folder = min(folders, key=os.path.getctime)
        shutil.rmtree(oldest_folder)

while capture.isOpened():
    current_time = datetime.now()
    folder_name = current_time.strftime('%Y%m%d_%H')
    folder_path = os.path.join(base_directory, folder_name)
    create_directory_if_not_exists(folder_path)
    
    file_name = current_time.strftime('%Y%m%d_%H%M%S.avi')
    file_path = os.path.join(folder_path, file_name)

    out = cv2.VideoWriter(file_path, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    frame_count = 0

    # while (time.time() - start_time) < record_duration:
    # 임시로 동영상의 프레임으로 1분 길이 체크
    while frame_count < total_frames_per_minute:
        ret, frame = capture.read()

        if not ret:
            capture.release()
            break

        out.write(frame)
        frame_count += 1
    
    out.release()

    # 폴더 용량 체크 후 삭제
    if get_directory_size(base_directory) > max_folder_size:
        remove_oldest_directory(base_directory)
        print()

cv2.destroyAllWindows()