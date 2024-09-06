import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

def img_1():
    print("1번 이미지")
    img_ori = cv2.imread('opencvDojang/data2/misson/01.png')
    img = cv2.imread('opencvDojang/data2/misson/01.png')
    
    if img is None or img_ori is None:
        sys.exit('Image load failed')
        
    bright_image = cv2.convertScaleAbs(img, alpha=1.1, beta=20)
        
    gray = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)
    
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    mask_inv = cv2.bitwise_not(mask)
    
    background = cv2.bitwise_and(img, img, mask=mask_inv)
    filtered_background = cv2.bilateralFilter(background, 25, 95, 95)
    
    building = cv2.bitwise_and(img, img, mask=mask)
    final_img = cv2.add(building, filtered_background)

    cv2.imshow('img_ori', img_ori)
    cv2.imshow('final_img', final_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def img_3():
    print("3번 이미지")
    img_ori = cv2.imread('opencvDojang/data2/misson/03.png')
    img = cv2.imread('opencvDojang/data2/misson/03.png')

    if img is None or img_ori is None:
        sys.exit('Image load failed')
        
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(img, -1, kernel)

    bright_image = cv2.convertScaleAbs(sharpened_image, alpha=1, beta=20)

    cv2.imshow('img_ori', img_ori)
    cv2.imshow('bright_image', bright_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
  
    
def img_5():
    print("5번 이미지")
    img_ori = cv2.imread('opencvDojang/data2/misson/05.png')
    img = cv2.imread('opencvDojang/data2/misson/05.png')
    
    if img is None or img_ori is None:
        sys.exit('Image load failed')
        
    # 샤프닝을 위한 커널 정의
    sharpen_kernel = np.array([[-1, -1, -1], 
                            [-1,  9, -1], 
                            [-1, -1, -1]])

    # 샤프닝 필터 적용
    sharpened_image = cv2.filter2D(img, -1, sharpen_kernel)
    
    cv2.imshow('img_ori', img_ori)
    cv2.imshow('img', sharpened_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    # img_1()
    # img_3()
    img_5()