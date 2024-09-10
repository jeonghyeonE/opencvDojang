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
    # image_resize = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    return image

orig_image = load_image(image_index)  # 첫 번째 이미지를 로드

# 이미지 크기
img_height, img_width = orig_image.shape[:2]

# 마우스로 클릭하고 드래그하여 사각형을 그리기 위한 변수 초기화
start_point = None
end_point = None
drawing = False
rectangle_complete = False
current_label = None
rectangles = []  # 여러 사각형을 저장할 리스트

# 마우스 콜백 함수 정의
def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, drawing, rectangle_complete

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True
        rectangle_complete = False

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False
        rectangle_complete = True
        rectangles.append((start_point, end_point))  # 사각형을 리스트에 추가
        start_point = None
        end_point = None

# 이미지에서 사각형 부분만 마스크로 처리하는 함수
def apply_mask(image_resize, rectangles):
    mask = np.zeros(image_resize.shape[:2], dtype=np.uint8)
    for rect in rectangles:
        cv2.rectangle(mask, rect[0], rect[1], 255, -1)
    masked_image = cv2.bitwise_and(image_resize, image_resize, mask=mask)
    return masked_image

# # 사각형 변환 처리 함수
# def warp_to_rectangle(image, start_point, end_point):
#     rect = (min(start_point[0], end_point[0]), min(start_point[1], end_point[1]),
#             max(start_point[0], end_point[0]), max(start_point[1], end_point[1]))
#     x1, y1, x2, y2 = rect
#     width = x2 - x1
#     height = y2 - y1
#     dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
#     src_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
#     M = cv2.getPerspectiveTransform(src_points, dst_points)
#     warped_image = cv2.warpPerspective(image, M, (width, height))
#     return warped_image

# yolo 변환
def convert_to_yolo_format(start_point, end_point, img_width, img_height):
    x_min, y_min = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
    x_max, y_max = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
    
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return (x_center, y_center, width, height)

# 좌표 저장 함수
def save_rectangle(image_index, rectangles, img_height, img_width, label):
    image_name = os.path.splitext(image_files[image_index])[0]
    points_file = os.path.join(image_folder, f'PointData/all_points.txt')

    with open(points_file, 'a') as f:
        f.write(f'Image: {image_name}\n')
        for rect in rectangles:
            yolo_format = convert_to_yolo_format(rect[0], rect[1], img_width, img_height)
            f.write(f'{label} : {yolo_format}\n')

# 윈도우 설정 및 마우스 콜백 함수 연결
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    image_copy = orig_image.copy()

    # 그려진 모든 사각형을 표시
    for rect in rectangles:
        cv2.rectangle(image_copy, rect[0], rect[1], (150, 150, 235), 2)

    # 현재 그리고 있는 사각형도 표시
    if drawing and start_point and end_point:
        cv2.rectangle(image_copy, start_point, end_point, (150, 150, 235), 2)

    cv2.imshow('Image', image_copy)

    # 마스크된 이미지를 업데이트
    if len(rectangles) > 0:
        masked_image = apply_mask(orig_image, rectangles)
        cv2.imshow('Masked Image', masked_image)

    key = cv2.waitKeyEx(30)

    # 's' 키를 누르면 현재 그린 모든 사각형을 저장
    if key == ord('s'):
        if current_label is not None:
            save_rectangle(image_index, rectangles, img_height, img_width, current_label)
        else:
            print('레이블을 선택해주세요.')

    # 'c' 키를 누르면 모든 사각형 초기화
    if key == ord('c'):
        rectangles = []  # 사각형 리스트를 초기화
        cv2.destroyWindow('Masked Image')  # 마스크 창도 초기화 시 닫기
        
    # 'd' 키를 누르면 가장 최근에 그린 사각형을 삭제
    if key == ord('d'):
        if rectangles:  # 사각형이 있을 때만 삭제
            rectangles.pop()  # 가장 최근의 사각형 삭제
            cv2.destroyWindow('Masked Image')  # 마스크 창 초기화 후 다시 그리기

    # 왼쪽 화살표 키를 누르면 이전 이미지
    if key == 0x250000:
        if image_index > 0:
            image_index -= 1
            orig_image = load_image(image_index)
            img_height, img_width = orig_image.shape[:2]
            rectangles = []  # 새로운 이미지로 변경 시 사각형 리스트 초기화
            current_label = None

    # 오른쪽 화살표 키를 누르면 다음 이미지
    if key == 0x270000:
        if image_index < len(image_files) - 1:
            image_index += 1
            orig_image = load_image(image_index)
            img_height, img_width = orig_image.shape[:2]
            rectangles = []  # 새로운 이미지로 변경 시 사각형 리스트 초기화
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
