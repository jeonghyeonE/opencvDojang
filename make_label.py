import cv2
import numpy as np
import os

# 이미지 파일을 불러올 폴더 경로 설정
image_folder = 'opencvDojang/data2/'  # 이미지 폴더 경로 설정

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_index = 0  # 현재 표시할 이미지의 인덱스

# 현재 선택된 이미지 불러오기
def load_image(image_index):
    image_path = os.path.join(image_folder, image_files[image_index])
    image = cv2.imread(image_path)
    image_resize = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    return image_resize

orig_image = load_image(image_index)  # 첫 번째 이미지를 로드

# 이미지 크기
img_height, img_width = orig_image.shape[:2]
circle_radius = 10  # 원의 반지름 설정

# 초기 다각형의 좌표 설정 (예: 사각형)
points = np.array([
    [int(img_width * 0.25), int(img_height * 0.25)],  # 좌상단
    [int(img_width * 0.75), int(img_height * 0.25)],  # 우상단
    [int(img_width * 0.75), int(img_height * 0.75)],  # 우하단
    [int(img_width * 0.25), int(img_height * 0.75)]   # 좌하단
], dtype=np.int32)

selected_point = None  # 현재 선택된 포인트
current_label = None  # 선택된 레이블 (1 또는 2)

# 마우스 콜백 함수 정의
def mouse_callback(event, x, y, flags, param):
    global points, selected_point

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, point in enumerate(points):
            if np.linalg.norm(point - np.array([x, y])) < 10:  # 근처 좌표일 때
                selected_point = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and selected_point is not None:
        points[selected_point] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = None

# 이미지에서 다각형 부분만 마스크로 처리하는 함수
def apply_mask(image_resize, points):
    mask = np.zeros(image_resize.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    masked_image = cv2.bitwise_and(image_resize, image_resize, mask=mask)
    return masked_image

# 사각형 변환 처리 함수
def warp_to_rectangle(image, points):
    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    src_points = points.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, M, (w, h))
    return warped_image

# yolo 변환
def convert_to_yolo_format(bbox, img_width, img_height):
    # bbox: [(x_min, y_min), (x_max, y_max)]
    x_min, y_min = bbox[0]
    x_max, y_max = bbox[2]
    
    # 중심 좌표 계산
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    
    # 너비와 높이 계산
    width = x_max - x_min
    height = y_max - y_min
    
    # YOLO 형식으로 정규화
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return (x_center, y_center, width, height)

# 좌표 저장 함수
def save_points(image_index, points, img_height, img_width, label):
    # 이미지 파일명에서 확장자를 제거하고 파일명만 추출
    image_name = os.path.splitext(image_files[image_index])[0]
    # 좌표를 저장할 파일 경로 생성
    points_file = os.path.join(image_folder, f'PointData/all_points.txt')
    points_lst = []
    
    for point in points:
            points_lst.append(point.tolist())
    
    yolo_format = convert_to_yolo_format(points_lst, img_width, img_height)
    
    with open(points_file, 'a') as f:
        # 이미지 파일명에서 확장자를 제거하고 파일명만 추출
        image_name = os.path.splitext(image_files[image_index])[0]
        # for point in points:
        #     points_lst.append(point.tolist())
        f.write(f'Image: {image_name}\n')  # 이미지 파일명 기록
        f.write(f'{label} : {yolo_format}\n') # 좌표값 저장

# 윈도우 설정 및 마우스 콜백 함수 연결
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    # 복사된 이미지를 이용해 다각형을 그리기
    image_copy = orig_image.copy()

    for point in points:
        cv2.circle(image_copy, tuple(point), 10, (150, 150, 235), 1)

    cv2.polylines(image_copy, [points], isClosed=True, color=(192, 192, 235), thickness=2)
    masked_image = apply_mask(orig_image, points)

    cv2.imshow('Image', image_copy)
    cv2.imshow('Masked Image', masked_image)

    # key1 = cv2.waitKeyEx(1) & 0xFF
    key = cv2.waitKeyEx(30)

    # 's' 키를 누르면 다각형 영역을 사각형으로 변환하여 새 창에 표시
    if key == ord('s'):
        if current_label is not None:  # 레이블이 선택되었을 때만 저장 가능
            if len(points) >= 4:
                img_height, img_width = orig_image.shape[:2]
                warped_image = warp_to_rectangle(orig_image, points)
                cv2.imshow('Warped Image', warped_image)
                save_points(image_index, points, img_height, img_width, current_label)  # 좌표 저장 함수 호출
        else:
            print('레이블을 선택해주세요.')

    # 왼쪽 화살표 키를 누르면 이전 이미지
    if key ==0x250000:  # 왼쪽 방향키
        if image_index > 0:
            image_index -= 1
            orig_image = load_image(image_index)
            # 이미지 크기
            img_height, img_width = orig_image.shape[:2]
            circle_radius = 10  # 원의 반지름 설정
            # 초기 다각형의 좌표 설정 (예: 사각형)
            points = np.array([
                [int(img_width * 0.25), int(img_height * 0.25)],  # 좌상단
                [int(img_width * 0.75), int(img_height * 0.25)],  # 우상단
                [int(img_width * 0.75), int(img_height * 0.75)],  # 우하단
                [int(img_width * 0.25), int(img_height * 0.75)]   # 좌하단
            ], dtype=np.int32)
            current_label = None

    # 오른쪽 화살표 키를 누르면 다음 이미지
    if key ==0x270000:  # 오른쪽 방향키
        if image_index < len(image_files) - 1:
            image_index += 1
            orig_image = load_image(image_index)
            # 이미지 크기
            img_height, img_width = orig_image.shape[:2]
            circle_radius = 10  # 원의 반지름 설정
            # 초기 다각형의 좌표 설정 (예: 사각형)
            points = np.array([
                [int(img_width * 0.25), int(img_height * 0.25)],  # 좌상단
                [int(img_width * 0.75), int(img_height * 0.25)],  # 우상단
                [int(img_width * 0.75), int(img_height * 0.75)],  # 우하단
                [int(img_width * 0.25), int(img_height * 0.75)]   # 좌하단
            ], dtype=np.int32)
            current_label = None
            
    # 1 또는 2 키로 레이블을 선택
    if key == ord('1'):
        current_label = 1
        print("레이블 1이 선택되었습니다.")
    elif key == ord('2'):
        current_label = 2
        print("레이블 2가 선택되었습니다.")

    # 'q' 키를 누르면 종료
    if key == ord('q'):
        break

cv2.destroyAllWindows()
