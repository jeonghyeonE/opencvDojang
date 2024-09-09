import cv2
import numpy as np

# 마우스 이벤트 처리 함수
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭 시
        print(f"왼쪽 버튼 클릭 좌표: ({x}, {y})")
        cv2.circle(img, (x, y), 20, (255, 0, 0), 1)  # 클릭한 곳에 원을 그립니다.
        cv2.imshow('image', img)

# 빈 이미지 생성
img = np.ones((512,512,3), np.uint8) * 255
# img = cv2.imread('path_to_your_image.jpg')  # 이미지 파일을 읽어옵니다.
cv2.imshow('image', img)

# 'image' 창에 마우스 콜백 함수 연결
cv2.setMouseCallback('image', mouse_callback)

# ESC 키를 누르면 종료
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키는 ASCII 코드로 27
        break

cv2.destroyAllWindows()
