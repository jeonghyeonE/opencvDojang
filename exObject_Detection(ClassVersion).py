import numpy as np
import cv2
import os

class ImageProcessor:
    def __init__(self, ori_data_path, multi_data_path, rotate=True, flip=True, brightness=True, zoom=True, shift=True):
        # 이미지 경로 초기화
        self.ori_data_path = ori_data_path
        self.multi_data_path = multi_data_path

        # 최상위 폴더가 없으면 생성
        if not os.path.exists(self.multi_data_path):
            os.makedirs(self.multi_data_path)

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

    def image_resize(self, fileName):
        # 이미지 불러오기
        file = os.path.join(self.ori_data_path, fileName)
        image = cv2.imread(file)
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        return image_resized

    def adjust_brightness(self, image, fileName, subfolder):
        if not self.brightness_enabled:
            return

        # 밝기 조절
        adjusted_dark = cv2.convertScaleAbs(image, alpha=0.8, beta=0)
        adjusted_bright = cv2.convertScaleAbs(image, alpha=1.2, beta=0)

        # 파일 저장
        cv2.imwrite(f'{subfolder}/{fileName}_dark.jpg', adjusted_dark)
        cv2.imwrite(f'{subfolder}/{fileName}_bright.jpg', adjusted_bright)

    def flip_image(self, image, fileName, subfolder):
        if not self.flip_enabled:
            return

        # 좌우/상하/좌우상하 반전
        h_flip = cv2.flip(image, 1)
        v_flip = cv2.flip(image, 0)
        hv_flip = cv2.flip(image, -1)

        # 파일 저장
        cv2.imwrite(f'{subfolder}/{fileName}_hflip.jpg', h_flip)
        cv2.imwrite(f'{subfolder}/{fileName}_vflip.jpg', v_flip)
        cv2.imwrite(f'{subfolder}/{fileName}_hvflip.jpg', hv_flip)

    def rotate_image(self, image, angle, fileName, subfolder):
        if not self.rotate_enabled:
            return

        # 이미지 회전
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(f'{subfolder}/{fileName}_rotated_{angle}.jpg', rotated)

    def zoom_image(self, image, scale, fileName, subfolder):
        if not self.zoom_enabled:
            return

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

        cv2.imwrite(f'{subfolder}/{fileName}_zoom_{scale}.jpg', zoomed)

    def shift_image(self, image, x_shift, y_shift, fileName, subfolder):
        if not self.shift_enabled:
            return

        # 이미지 이동 (쉬프트)
        h, w = image.shape[:2]
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(f'{subfolder}/{fileName}_shift_{x_shift}_{y_shift}.jpg', shifted)

    def process_images(self):
        # 폴더 내 모든 이미지 처리
        for fileName in os.listdir(self.ori_data_path):
            # 파일명에서 확장자를 제외한 이름을 추출
            file_base_name = fileName.split('.')[0]

            # 파일 이름 기반 폴더 생성
            subfolder = self.create_folder(file_base_name)

            # 이미지 리사이즈
            image = self.image_resize(fileName)

            # 회전
            if self.rotate_enabled:
                for angle in range(0, 360, 20):
                    self.rotate_image(image, angle, file_base_name, subfolder)

            # 상하좌우 반전
            self.flip_image(image, file_base_name, subfolder)

            # 밝기 조절
            self.adjust_brightness(image, file_base_name, subfolder)

            # 줌인, 줌아웃
            self.zoom_image(image, 1.2, file_base_name, subfolder)
            self.zoom_image(image, 0.8, file_base_name, subfolder)

            # 이미지 이동
            self.shift_image(image, 30, 0, file_base_name, subfolder)
            self.shift_image(image, -30, 0, file_base_name, subfolder)
            self.shift_image(image, 0, 20, file_base_name, subfolder)
            self.shift_image(image, 0, -20, file_base_name, subfolder)


if __name__ == '__main__':
    # 경로 설정
    ori_data_path = os.path.join(os.getcwd(), 'opencvDojang/data3/data_ori')
    multi_data_path = os.path.join(os.getcwd(), 'opencvDojang/data3/data_multi')

    # 사용자 옵션에 따라 기능 활성화/비활성화
    processor = ImageProcessor(ori_data_path, multi_data_path, rotate=True, flip=True, brightness=True, zoom=True, shift=True)
    processor.process_images()
