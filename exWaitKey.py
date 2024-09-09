import cv2
import numpy as np

# 초기 원의 좌표 및 크기 설정
circle_radius = 30
circle_position = [250, 250]  # (x, y)
circle_color = (255, 0, 0)  # 파란색
dirction = 0

img = 255 * np.ones((500, 500, 3), dtype=np.uint8)  # 이미지가 없을 경우 빈 흰색 배경

cv2.imshow('image', img)

while True:
    temp_img = img.copy()  # 원을 그리고 화면을 업데이트하기 위해 이미지를 복사
    
    # 원을 그리기
    cv2.circle(temp_img, (circle_position[0], circle_position[1]), circle_radius, circle_color, 1)
    cv2.imshow('image', temp_img)
    
    key = cv2.waitKeyEx(30)
    print(key)
    
    # 종료 조건
    if key == 27: #ESC
        break
    # right key
    elif key ==0x270000:
        direction =0
        circle_position[0]+=10
    # down key
    elif key ==0x280000:
        direction =1
        circle_position[1]+=10
    # left key
    elif key ==0x250000:
        direction =2
        circle_position[0]-=10
    # up key
    elif key ==0x260000:
        direction =3
        circle_position[1]-=10

cv2.destroyAllWindows()
