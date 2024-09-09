import cv2, sys
import numpy as np

pt1 = (0,0)
pt2 = (0,0)
pt_lst = []


def mouse_callback(event, x, y, flags, param):
    img = param[0]
    global pt1, pt2, pt_lst
    
    if event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 버튼 클릭 시
        cv2.circle(img, (x, y), 50, (255, 0, 0), 1)  # 클릭한 곳에 원을 그립니다.
    
    if flags & cv2.EVENT_FLAG_SHIFTKEY:
        if event==cv2.EVENT_LBUTTONDOWN:
            pt_lst.append(list((x,y)))
    else:
        if len(pt_lst) >= 3:
            polygon = np.array(pt_lst)
            cv2.polylines(img, [polygon], isClosed = True, color = (255,0,0))
            pt_lst = []
        
    if event==cv2.EVENT_LBUTTONDOWN:
        pt1 = (x,y)
    elif event==cv2.EVENT_LBUTTONUP:
        pt2 = (x,y)
        cv2.rectangle(img, pt1, pt2, (255,0,0), 1)    
        
    # 그린 화면을 업데이트
    cv2.imshow('img',img)
    
# 이미지 저장하기
def save_image(img, filename="opencvDojang/data2/result.png"):
    cv2.imwrite(filename, img)
    
    
# 이미지 생성
img = np.ones((512,512,3), np.uint8) * 255
cv2.namedWindow('img')

cv2.setMouseCallback('img', mouse_callback, [img])

while True:
    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q'를 누르면 종료
        break
    elif key == ord('s'):  # 's'를 누르면 이미지 저장
        save_image(img, 'opencvDojang/data2/result.png')
        
cv2.destroyAllWindows()