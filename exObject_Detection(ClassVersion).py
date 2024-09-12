import numpy as np
import cv2
import os

class ImageProcessor:
    def __init__(self, ori_data_path, multi_data_path, rotate=True, flip=True, brightness=True, zoom=True, shift=True):
        # 이미지 경로 초기화
        self.ori_data_path = ori_data_path
        self.multi_data_path = multi_data_path

        # 기능별 실행 여부 설정
        self.rotate_enabled = rotate
        self.flip_enabled = flip
        self.brightness_enabled = brightness
        self.zoom_enabled = zoom
        self.shift_enabled = shift

    def create_folder(self, file_name):
        # 파일 이름 기반 폴더 생성
        subfolder = os.path.join(self.multi_data_path, file_name)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        return subfolder
    
    def save_image(self, image, file_name, operation, subfolder):
        file_name = file_name.split('.')[0]
        file_path = os.path.join(subfolder, f'{file_name}_{operation}.jpg')
        cv2.imwrite(file_path, image)

    def image_resize(self, fileName):
        # 이미지 불러오기
        file = os.path.join(self.ori_data_path, fileName)
        image = cv2.imread(file)
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        return image_resized

    def adjust_brightness(self, image, fileName, subfolder):
        # 밝기 조절
        adjusted_dark = cv2.convertScaleAbs(image, alpha=0.8, beta=0)
        adjusted_bright = cv2.convertScaleAbs(image, alpha=1.2, beta=0)

        # 파일 저장
        self.save_image(adjusted_dark, fileName, f'dark', subfolder)
        self.save_image(adjusted_bright, fileName, f'bright', subfolder)

    def flip_image(self, image, fileName, subfolder):
        # 좌우/상하/좌우상하 반전
        h_flip = cv2.flip(image, 1)
        v_flip = cv2.flip(image, 0)
        hv_flip = cv2.flip(image, -1)

        # 파일 저장
        self.save_image(h_flip, fileName, f'h_flip', subfolder)
        self.save_image(v_flip, fileName, f'v_flip', subfolder)
        self.save_image(hv_flip, fileName, f'hv_flip', subfolder)

    def rotate_image(self, image, angle, fileName, subfolder):
        # 이미지 회전
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        self.save_image(rotated, fileName, f'rotated_{angle}', subfolder)

    def zoom_image(self, image, scale, fileName, subfolder):
        # 줌인/줌아웃
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if scale > 1:  # 줌인
            start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
            zoomed = zoomed[start_h:start_h + h, start_w:start_w + w]
        else:  # 줌아웃
            pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
            zoomed = cv2.copyMakeBorder(zoomed, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REPLICATE)

        self.save_image(zoomed, fileName, f'zoom_{scale}', subfolder)

    def shift_image(self, image, x_shift, y_shift, fileName, subfolder):
        # 이미지 이동 (쉬프트)
        h, w = image.shape[:2]
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        self.save_image(shifted, fileName, f'shift_{x_shift}_{y_shift}', subfolder)

    def process_images(self):
        # 폴더 내 모든 이미지 처리
        for fileName in os.listdir(self.ori_data_path):
            # 파일 이름(클래스) 기반 폴더 생성
            subfolder = self.create_folder(fileName.split('_')[0])

            # 이미지 리사이즈
            image = self.image_resize(fileName)

            # 회전
            if self.rotate_enabled:
                for angle in range(0, 360, 20):
                    self.rotate_image(image, angle, fileName, subfolder)

            # 상하좌우 반전
            if self.flip_enabled:
                self.flip_image(image, fileName, subfolder)

            # 밝기 조절
            if self.brightness_enabled:
                self.adjust_brightness(image, fileName, subfolder)

            # 줌인, 줌아웃
            if self.zoom_enabled:
                self.zoom_image(image, 1.2, fileName, subfolder)
                self.zoom_image(image, 0.8, fileName, subfolder)

            # 이미지 이동
            if self.shift_enabled:
                self.shift_image(image, 30, 0, fileName, subfolder)
                self.shift_image(image, -30, 0, fileName, subfolder)
                self.shift_image(image, 0, 20, fileName, subfolder)
                self.shift_image(image, 0, -20, fileName, subfolder)


if __name__ == '__main__':
    # 경로 설정
    ori_data_path = os.path.join(os.getcwd(), 'opencvDojang/data3/data_ori')
    multi_data_path = os.path.join(os.getcwd(), 'opencvDojang/data3/data_multi')

    # 사용자 옵션에 따라 기능 활성화/비활성화
    processor = ImageProcessor(ori_data_path, multi_data_path, rotate=True, flip=True, brightness=True, zoom=True, shift=True)
    processor.process_images()
