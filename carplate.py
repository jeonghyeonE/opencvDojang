import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

video_file = 'opencvDojang/data/carplate2.mp4'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cap = cv2.VideoCapture(video_file)  # 동영상 파일 열기

if not cap.isOpened():  # 파일이 열리지 않을 경우
    print('video open failed')
    sys.exit()

while True:
    ret, frame = cap.read()  # 프레임 읽기
    height, width, channel = frame.shape  # 프레임의 높이, 너비, 채널 정보
    
    if not ret:  # 읽어온 프레임이 없으면 종료
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임을 회색으로 변환
    blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)  # 가우시안 블러로 노이즈 줄이기
    
    thresh = cv2.adaptiveThreshold(  # 블러 처리된 이미지를 흑백으로 변환
        blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )
    
    contours, _ = cv2.findContours(  # 윤곽선 찾기
        thresh, 
        mode=cv2.RETR_LIST, 
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 빈 이미지 생성
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))  # 윤곽선을 흰색으로 그림
    
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 빈 이미지 초기화
    contours_dict = []
    
    for contour in contours:  # 윤곽선 중 사각형 찾기
        x, y, w, h = cv2.boundingRect(contour)  # 사각형 경계 계산
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)  # 사각형 그리기
    
        contours_dict.append({  # 사각형 정보 저장
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2),
        })

    # 필터링 조건 정의
    MIN_AREA = 30
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0
    
    possible_contours = []
    
    cnt = 0
    
    for d in contours_dict:
        area = d['w'] * d['h']  # 영역 계산
        ratio = d['w'] / d['h']  # 가로 세로 비율 계산
        
        if area > MIN_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:  # 조건에 맞는 윤곽선만 저장
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
            
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 빈 이미지 초기화
    
    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)  # 가능한 윤곽선을 다시 그리기
        
    # 매칭할 윤곽선의 조건 설정
    MAX_DIAG_MULTIPLYER = 5
    MAX_ANGLE_DIFF = 12.0
    MAX_AREA_DIFF = 0.5
    MAX_WIDTH_DIFF = 0.8
    MAX_HEIGHT_DIFF = 0.2
    MIN_N_MATCHED = 4
    
    def find_chars(contour_list):  # 윤곽선을 매칭하는 함수
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w']**2 + d1['h']**2)
                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))

                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:  # 조건에 맞는 윤곽선 매칭
                    matched_contours_idx.append(d2['idx'])

            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []

            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx
    
    result_idx = find_chars(possible_contours)  # 윤곽선 찾기
    
    matched_result = []
    
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
        
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 빈 이미지 초기화
    
    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)  # 매칭된 윤곽선 그리기
    
    # 번호판 후보 영역 설정
    PLATE_WIDTH_PADDING = 1.3
    PLATE_HEIGHT_PADDING = 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10
    
    plate_videos = []
    plate_infos = []
    
    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])  # 문자들을 x좌표 기준으로 정렬
    
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        
        sum_height = 0
        
        for d in sorted_chars:
            sum_height += d['h']
            
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
    
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        video_rotated = cv2.warpAffine(thresh, M=rotation_matrix, dsize=(width, height))  # 회전된 이미지 생성
        video_cropped = cv2.getRectSubPix(
            video_rotated,
            patchSize = (int(plate_width), int(plate_height)),
            center = (int(plate_cx), int(plate_cy))
        )
        
        if video_cropped.shape[1] / video_cropped.shape[0] < MIN_PLATE_RATIO or video_cropped.shape[1] / video_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue
    
        plate_videos.append(video_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
    
    longest_idx, longest_text = -1, 0
    plate_chars = []

    for i, plate_video in enumerate(plate_videos):
        plate_video = cv2.resize(plate_video, dsize=(0, 0), fx=1.6, fy=1.6)  # 이미지 확대
        _, plate_video = cv2.threshold(plate_video, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(plate_video, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)  # 윤곽선 다시 찾기

        plate_min_x, plate_min_y = plate_video.shape[1], plate_video.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            ratio = w / h

            if area > MIN_AREA \
            and w > MIN_WIDTH and h > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:  # 조건에 맞는 윤곽선만 추출
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h

        video_result = plate_video[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

        video_result = cv2.GaussianBlur(video_result, ksize=(3, 3), sigmaX=0)  # 블러 처리
        _, video_result = cv2.threshold(video_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        video_result = cv2.copyMakeBorder(video_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))  # 테두리 추가

        # chars = pytesseract.image_to_string(video_result, lang='kor', config='--psm 7 --oem 0') # 한글
        chars = pytesseract.image_to_string(video_result, lang='eng', config='--psm 1 -c preserve_interword_spaces=1')  # Tesseract로 문자 인식

        result_chars = ''
        has_digit = False
        for c in chars:
            # if ord('가') <= ord(c) <= ord('힣') or c.isdigit(): # 인식된 문자가 한글 또는 숫자인 경우
            if ('A' <= c <= 'Z' or 'a' <= c <= 'z') or c.isdigit():  # 인식된 문자가 영문자 또는 숫자인 경우
                if c.isdigit():
                    has_digit = True
                result_chars += c

        plate_chars.append(result_chars)

        if has_digit and len(result_chars) > longest_text:  # 숫자가 포함된 가장 긴 텍스트 저장
            longest_idx = i
    
    info = plate_infos[longest_idx]  # 최종 선택된 번호판 영역 정보
    chars = plate_chars[longest_idx]  # 최종 선택된 문자

    video_out = frame.copy()

    cv2.rectangle(video_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(0,0,255), thickness=2)  # 번호판 영역 표시
    cv2.putText(
        video_out,
        chars,
        org=(info['x'], info['y'] - 20),  # 번호판 위에 인식된 문자 표시
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1.1,
        color = (0, 255, 255),
        thickness = 2,
    )
    
    cv2.imshow('video_cropped', video_out)  # 동영상 출력
    
    if cv2.waitKey(60) == 27:  # ESC 키를 누르면 종료
        break
    

cap.release()  # 리소스 해제
cv2.destroyAllWindows()  # 창 닫기
