import cv2
import numpy as np

# 1. 이미지 불러오기
image_path = 'opencvDojang/data2/book.jpg'  # 이미지 경로 설정
image = cv2.imread(image_path)
image_resize = cv2.resize(image,(0,0),fx=0.5,fy=0.5)
orig_image = image_resize.copy()  # 원본 이미지를 유지

# 2. 초기 다각형의 좌표 설정 (예: 사각형)
points = np.array([[100, 100], [400, 100], [400, 400], [100, 400]], dtype=np.int32)
selected_point = None  # 현재 선택된 포인트

# 3. 마우스 콜백 함수 정의
def mouse_callback(event, x, y, flags, param):
    global points, selected_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # 마우스 좌표 근처에 있는 포인트를 선택 (포인트 근처의 원 안에 들어왔을 때만 선택 가능)
        for i, point in enumerate(points):
            if np.linalg.norm(point - np.array([x, y])) < 10:  # 근처 좌표일 때
                selected_point = i
                break

    elif event == cv2.EVENT_MOUSEMOVE and selected_point is not None:
        # 선택된 포인트를 이동
        points[selected_point] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        # 포인트 선택 해제
        selected_point = None

# 4. 이미지에서 다각형 부분만 마스크로 처리하는 함수
def apply_mask(image_resize, points):
    mask = np.zeros(image_resize.shape[:2], dtype=np.uint8)  # 이미지 크기와 같은 마스크 생성
    cv2.fillPoly(mask, [points], 255)  # 다각형 내부를 흰색으로 채움
    masked_image = cv2.bitwise_and(image_resize, image_resize, mask=mask)  # 마스크를 사용해 이미지 오리기
    return masked_image

# 5. 윈도우 설정 및 마우스 콜백 함수 연결
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    # 복사된 이미지를 이용해 다각형을 그리기
    image_copy = orig_image.copy()
    
    # 각 코너에 원 그리기 (마우스로 이동할 수 있음을 시각적으로 표시)
    for point in points:
        cv2.circle(image_copy, tuple(point), 10, (150, 150, 235), 1)  # 원

    # 다각형을 그리기
    cv2.polylines(image_copy, [points], isClosed=True, color=(192, 192, 235), thickness=2)

    # 현재 다각형 영역을 마스크 처리한 이미지 생성
    masked_image = apply_mask(orig_image, points)

    # 원본 이미지 창과 오려낸 이미지를 별도의 창에 표시
    cv2.imshow('Image', image_copy)  # 원본 이미지에 다각형과 코너 원이 있는 모습
    cv2.imshow('Masked Image', masked_image)  # 오려낸 이미지

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
