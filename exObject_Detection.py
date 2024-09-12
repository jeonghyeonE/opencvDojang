import numpy as np
import cv2, os

def image_resize(ori_data_path, fileName):
    # 이미지 불러오기
    file = os.path.join(ori_data_path, fileName)
    image = cv2.imread(file)
    image_resize = cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA)

    # 이미지 확인
    # cv2.imshow('image_resize', image_resize)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return image_resize



def create_folder(multi_data_path, file_name):
        # 파일 이름 기반 폴더 생성
        subfolder = os.path.join(multi_data_path, file_name)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        return subfolder



def adjust_brightness(image, fileName, subfolder):
    # 밝기 조절: 이미지 픽셀 값에 factor를 곱함
    adjusted_dark = cv2.convertScaleAbs(image, alpha=0.8, beta=0)
    adjusted_bright = cv2.convertScaleAbs(image, alpha=1.2, beta=0)

    cv2.imwrite(f'{subfolder}/{fileName}_dark.jpg', adjusted_dark)
    cv2.imwrite(f'{subfolder}/{fileName}_bright.jpg', adjusted_bright)



def flip_image(image, fileName, subfolder):
    h_flip = cv2.flip(image, 1)  # 1: 좌우 반전
    v_flip = cv2.flip(image, 0)  # 0: 상하 반전
    hv_flip = cv2.flip(image, -1) # -1: 좌우상하 반전

    cv2.imwrite(f'{subfolder}/{fileName}_hflip.jpg', h_flip)
    cv2.imwrite(f'{subfolder}/{fileName}_vflip.jpg', v_flip)
    cv2.imwrite(f'{subfolder}/{fileName}_hvflip.jpg', hv_flip)



def rotate_image(image, angle, fileName, subfolder):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 회전 행렬 계산
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(f'{subfolder}/{fileName}_rotated_{angle}.jpg', rotated)



def zoom_image(image, scale, fileName, subfolder):
    # 이미지 크기 조절 (줌인, 줌아웃)
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale > 1:  # 줌인
        # 중앙 부분을 잘라냄
        start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
        zoomed = zoomed[start_h:start_h + h, start_w:start_w + w]
    else:  # 줌아웃
        # 패딩 추가
        pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
        zoomed = cv2.copyMakeBorder(zoomed, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REPLICATE)

    cv2.imwrite(f'{subfolder}/{fileName}_zoom_{scale}.jpg', zoomed)



def shift_image(image, x_shift, y_shift, fileName, subfolder):
    # 이미지 이동 (쉬프트)
    h, w = image.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    cv2.imwrite(f'{subfolder}/{fileName}_shift_{x_shift}_{y_shift}.jpg', shifted)



def main():
    ori_data_path = os.path.join(os.getcwd(),'opencvDojang/data3/data_ori')
    multi_data_path = os.path.join(os.getcwd(),'opencvDojang/data3/data_multi')

    # 최상위 폴더가 없으면 생성
    if not os.path.exists(multi_data_path):
        os.makedirs(multi_data_path)

    for fileName in os.listdir(ori_data_path):
        # 파일명에서 확장자를 제외한 이름을 추출
        file_base_name = fileName.split('.')[0]

        # 파일 이름 기반 폴더 생성
        subfolder = create_folder(multi_data_path, file_base_name)

        image = image_resize(ori_data_path, fileName) # 이미지 원본 파일 경로 입력

        # 회전시키기
        for angle in range(0, 360, 20):
            rotate_image(image, angle, file_base_name, subfolder)

        # 상하좌우 반전
        flip_image(image, file_base_name, subfolder)

        # 밝기를 조절
        adjust_brightness(image, file_base_name, subfolder)

        # 줌인, 줌아웃
        zoom_image(image, 1.2, file_base_name, subfolder)
        zoom_image(image, 0.8, file_base_name, subfolder)

        # 이미지 이동
        shift_image(image, 30, 0, file_base_name, subfolder)
        shift_image(image, -30, 0, file_base_name, subfolder)
        shift_image(image, 0, 20, file_base_name, subfolder)
        shift_image(image, 0, -20, file_base_name, subfolder)

if __name__ == '__main__':
    main()