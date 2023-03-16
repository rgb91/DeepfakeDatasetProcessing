"""
Converts video_to_frames_all_folders including align_and_crop_face_from_frames
"""
import os
from csv import writer, DictWriter
from pprint import pprint

import cv2
import shutil
import threading
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from deepface import DeepFace
import logging

# logging.basicConfig(level=logging.INFO, filename='./FFpp_bounding_box_calculation.log')
logging.basicConfig(level=logging.INFO, filename='./FFpp_crop_resize_errors.log')

# KPS = 5
COMPRESSION = 'c23'  # 'raw' takes up the whole HDD
IMAGE_EXTENSION = '.jpg'  # .png takes way more space
FACE_RESOLUTION = 224
VIDEO_HOME = r'/data/DATASETS/FaceForensicsPP'
TEMP_HOME = rf'/data/DATASETS/FaceForensicsPP-processed/FFpp-{COMPRESSION}-frames'
CROPPED_FRAMES_HOME = rf'/data/DATASETS/FFpp-{COMPRESSION}-{FACE_RESOLUTION}'
DATA_CONFIG_HOME = rf'/data/DATASETS/FFpp-{COMPRESSION}-{FACE_RESOLUTION}-config'
CLASSES = ['manipulated_sequences', 'original_sequences']
# DATASETS = [['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures'], ['youtube']]
DATASETS = [['FaceSwap', 'NeuralTextures'], ['youtube']]
CLASSES_NAME_MAP = {
    'Deepfakes': 'fake_DF',
    'Face2Face': 'fake_F2F',
    'FaceShifter': 'fake_FSh',
    'FaceSwap': 'fake_FS',
    'NeuralTextures': 'fake_NT',
    'youtube': 'real'
}


def get_face_area(video_path, df):
    # row = df.loc[df['video_path'] == video_path]
    # return int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max']), int(row['w']), int(row['h'])
    row = df.loc[df[0] == video_path]
    return int(row[1]), int(row[2]), int(row[5]), int(row[6]), int(row[3]), int(row[4])


def get_train_or_test(video_name):
    if len(video_name.split('_')) == 1:  # 'real'
        test_real = pd.read_csv(os.path.join(DATA_CONFIG_HOME, 'test_real.csv'), header=None)[0].values.tolist()
        if int(video_name) in test_real:
            return 'test'
        return 'train'
    else:  # 'fake'
        test_fake = pd.read_csv(os.path.join(DATA_CONFIG_HOME, 'test_fake.csv'), header=None)[0].values.tolist()
        if video_name in test_fake:
            return 'test'
        return 'train'


def empty_file(filepath):
    return os.path.isfile(filepath) and os.path.getsize(filepath) > 0


def get_bounding_box_all_videos():
    kps = 2

    try:
        done_list = pd.read_csv(os.path.join(DATA_CONFIG_HOME, 'area_done_list.csv'), header=None).values
        done_list = [item for t_list in done_list for item in t_list]
    except pd.errors.EmptyDataError:
        done_list = []

    for i, c in enumerate(CLASSES):
        for d in DATASETS[i]:
            source_dir = os.path.join(VIDEO_HOME, c, d, COMPRESSION, 'videos')
            video_list = [os.path.join(source_dir, v_name) for v_name in os.listdir(source_dir)]
            for video_path in tqdm(video_list, desc=d):
                if video_path in done_list:
                    continue
                video_cap = cv2.VideoCapture(video_path)
                success, image = video_cap.read()
                x_min, y_min = 9999, 9999
                w_max, h_max = -1, -1
                x_max, y_max = -1, -1
                fps = round(video_cap.get(cv2.CAP_PROP_FPS))
                count, hop = 0, round(fps // kps)  # hop = round(fps/kps) where kps is keyframes per sec.
                err_count = 0
                while success:
                    if count % hop == 0:
                        try:
                            face_image_obj = DeepFace.extract_faces(img_path=image,
                                                                    detector_backend='dlib',
                                                                    enforce_detection=True)
                            face_image_ob = face_image_obj[0]  # 'face', 'facial_area', 'confidence'
                            face_area = face_image_ob['facial_area']
                            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                            x_min, y_min, w_max, h_max = min(x_min, x), min(y_min, y), max(w_max, w), max(h_max, h)
                            x_max, y_max = max(x_max, x), max(y_max, y)
                            # Debug only
                            # crop_img = image[y:y + h, x:x + w]
                            # cv2.imwrite(f'./FFpp_frames_dummy/{video_name}/{count:05d}.jpg', crop_img)
                            # cv2.waitKey(0)
                        except ValueError:
                            err_count += 1
                    success, image = video_cap.read()
                    count += 1
                video_cap.release()
                if err_count > 0:
                    logging.info(f'Face not found. Count: {err_count}, Video: {video_path}.')
                crop_info = {
                    'video_path': video_path,
                    'x_min': x_min,
                    'y_min': y_min,
                    'w': w_max,
                    'h': h_max,
                    'x_max': x_max,
                    'y_max': y_max,
                }
                with open(os.path.join(DATA_CONFIG_HOME, 'FFpp_face_area.csv'), 'a') as fe:
                    d_writer = DictWriter(fe, fieldnames=crop_info.keys())
                    d_writer.writerow(crop_info)
                with open(os.path.join(DATA_CONFIG_HOME, 'area_done_list.csv'), 'a') as fd:
                    csv_writer = writer(fd)
                    csv_writer.writerow([video_path])
    pass


def extract_and_crop_all_videos():
    area_df = pd.read_csv(os.path.join(DATA_CONFIG_HOME, 'FFpp_face_area.csv'), header=None)

    try:  # to keep track of already processed files
        done_list = pd.read_csv(os.path.join(DATA_CONFIG_HOME, 'crop_done_list.csv'), header=None).values
        done_list = [item for t_list in done_list for item in t_list]
    except pd.errors.EmptyDataError:
        done_list = []

    for i, c in enumerate(CLASSES):
        for d in DATASETS[i]:
            source_dir = os.path.join(VIDEO_HOME, c, d, COMPRESSION, 'videos')
            video_list = [os.path.join(source_dir, v_name) for v_name in os.listdir(source_dir)]

            for video_path in tqdm(video_list, desc=d):
                if video_path in done_list:
                    continue
                video_name = os.path.basename(video_path).split('.')[0]
                train_or_test = get_train_or_test(video_name)
                destination_dir = os.path.join(CROPPED_FRAMES_HOME, train_or_test, CLASSES_NAME_MAP[d], video_name)
                Path(destination_dir).mkdir(parents=True, exist_ok=True)

                video_cap = cv2.VideoCapture(video_path)
                success, image = video_cap.read()
                count, err_count = 0, 0
                x_min, y_min, x_max, y_max, w, h = get_face_area(video_path, area_df)
                while success:
                    try:
                        crop_image = image[y_min:y_max + h, x_min:x_max + w]
                        crop_image = cv2.resize(crop_image, (FACE_RESOLUTION, FACE_RESOLUTION))
                        frame_path = os.path.join(destination_dir, f'{count:05d}.jpg')
                        cv2.imwrite(frame_path, crop_image)
                    except Exception as e:
                        err_count += 1
                    success, image = video_cap.read()
                    count += 1
                video_cap.release()
                if err_count > 0:
                    logging.info(f'Error Cropping/Resizing. Count: {err_count}, Video: {video_path}.')
                with open(os.path.join(DATA_CONFIG_HOME, 'crop_done_list.csv'), 'a') as fd:
                    csv_writer = writer(fd)
                    csv_writer.writerow([video_path])


def create_train_test_split_skeleton():
    Path(DATA_CONFIG_HOME).mkdir(parents=True, exist_ok=True)
    vids_f = os.listdir(os.path.join(TEMP_HOME, 'manipulated_sequences', 'NeuralTextures'))
    vids_f, split_loc = sorted(vids_f), int(len(vids_f) * 0.8)
    train_split_fake, test_split_fake = pd.Series(vids_f[:split_loc]), pd.Series(vids_f[split_loc:])
    train_split_fake.to_csv(os.path.join(DATA_CONFIG_HOME, 'train_fake.csv'), index=False, header=False)
    test_split_fake.to_csv(os.path.join(DATA_CONFIG_HOME, 'test_fake.csv'), index=False, header=False)

    vids_r = os.listdir(os.path.join(TEMP_HOME, 'original_sequences', 'youtube'))
    vids_r, split_loc = sorted(vids_r), int(len(vids_r) * 0.8)
    train_split_real, test_split_real = pd.Series(vids_r[:split_loc]), pd.Series(vids_r[split_loc:])
    train_split_real.to_csv(os.path.join(DATA_CONFIG_HOME, 'train_real.csv'), index=False, header=False)
    test_split_real.to_csv(os.path.join(DATA_CONFIG_HOME, 'test_real.csv'), index=False, header=False)

    # create folders (skeleton)
    for d in DATASETS[0]:
        for v in vids_f:
            train_or_test = 'train' if v in train_split_fake.values else 'test'
            destination_dir = os.path.join(CROPPED_FRAMES_HOME, train_or_test, CLASSES_NAME_MAP[d], v)
            Path(destination_dir).mkdir(parents=True, exist_ok=True)

    for d in DATASETS[1]:
        for v in vids_r:
            train_or_test = 'train' if v in train_split_real.values else 'test'
            destination_dir = os.path.join(CROPPED_FRAMES_HOME, train_or_test, CLASSES_NAME_MAP[d], v)
            Path(destination_dir).mkdir(parents=True, exist_ok=True)


def recalculate_bounding_box_missing_videos():
    """
        FaceSwap/575_603.mp4
        FaceSwap/112_892.mp4
        NeuralTextures/112_892.mp4
        youtube/112.mp4
     :return:
     """
    missing_videos = [
        r'/data/DATASETS/FaceForensicsPP/original_sequences/youtube/c23/videos/112.mp4',
        r'/data/DATASETS/FaceForensicsPP/manipulated_sequences/NeuralTextures/c23/videos/112_892.mp4',
        r'/data/DATASETS/FaceForensicsPP/manipulated_sequences/FaceSwap/c23/videos/112_892.mp4',
        r'/data/DATASETS/FaceForensicsPP/manipulated_sequences/FaceSwap/c23/videos/575_603.mp4'
    ]

    for video_path in missing_videos:
        video_cap = cv2.VideoCapture(video_path)
        success, image = video_cap.read()
        x_min, y_min = 9999, 9999
        w_max, h_max = -1, -1
        x_max, y_max = -1, -1
        fps = round(video_cap.get(cv2.CAP_PROP_FPS))
        count, hop = 0, round(fps // 2)  # hop = round(fps/kps) where kps is keyframes per sec.
        err_count = 0
        while success:
            if count % hop == 0:
                try:
                    face_image_obj = DeepFace.extract_faces(img_path=image,
                                                            detector_backend='dlib',
                                                            enforce_detection=True)
                    face_image_ob = face_image_obj[0]  # 'face', 'facial_area', 'confidence'
                    face_area = face_image_ob['facial_area']
                    x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                    if x >= 0 and y >= 0 and w >= 0 and h >= 0:
                        x_min, y_min, w_max, h_max = min(x_min, x), min(y_min, y), max(w_max, w), max(h_max, h)
                        x_max, y_max = max(x_max, x), max(y_max, y)
                    # Debug only
                    # crop_img = image[y:y + h, x:x + w]
                    # cv2.imwrite(f'./{count:05d}.jpg', crop_img)
                    # cv2.waitKey(0)
                    # exit()
                except ValueError:
                    err_count += 1
            success, image = video_cap.read()
            count += 1
        video_cap.release()
        if err_count > 0:
            print(f'Face not found. Count: {err_count}, Video: {video_path}.')
        crop_info = {
            'video_path': video_path,
            'x_min': x_min,
            'y_min': y_min,
            'w': w_max,
            'h': h_max,
            'x_max': x_max,
            'y_max': y_max,
        }
        print(video_path)
        pprint(crop_info)
        print()
        print()


def recrop_frames_missing_videos():
    """
        FaceSwap/575_603.mp4
        FaceSwap/112_892.mp4
        NeuralTextures/112_892.mp4
        youtube/112.mp4
     :return:
     """
    missing_videos = [
        r'/data/DATASETS/FaceForensicsPP/original_sequences/youtube/c23/videos/112.mp4',
        r'/data/DATASETS/FaceForensicsPP/manipulated_sequences/NeuralTextures/c23/videos/112_892.mp4',
        r'/data/DATASETS/FaceForensicsPP/manipulated_sequences/FaceSwap/c23/videos/112_892.mp4',
        r'/data/DATASETS/FaceForensicsPP/manipulated_sequences/FaceSwap/c23/videos/575_603.mp4'
    ]
    class_names = ['youtube', 'NeuralTextures', 'FaceSwap', 'FaceSwap']
    area_df = pd.read_csv(os.path.join(DATA_CONFIG_HOME, 'FFpp_face_area.csv'), header=None)

    for video_path, d in zip(missing_videos, class_names):
        video_name = os.path.basename(video_path).split('.')[0]
        train_or_test = get_train_or_test(video_name)
        destination_dir = os.path.join('./re-crop', train_or_test, CLASSES_NAME_MAP[d], video_name)
        Path(destination_dir).mkdir(parents=True, exist_ok=True)

        video_cap = cv2.VideoCapture(video_path)
        success, image = video_cap.read()
        count, err_count = 0, 0
        x_min, y_min, x_max, y_max, w, h = get_face_area(video_path, area_df)
        while success:
            try:
                crop_image = image[y_min:y_max + h, x_min:x_max + w]
                crop_image = cv2.resize(crop_image, (FACE_RESOLUTION, FACE_RESOLUTION))
                frame_path = os.path.join(destination_dir, f'{count:05d}.jpg')
                cv2.imwrite(frame_path, crop_image)
            except Exception as e:
                err_count += 1
            success, image = video_cap.read()
            count += 1
        video_cap.release()
        if err_count > 0:
            print(f'Error Cropping/Resizing. Count: {err_count}, Video: {video_path}.')


if __name__ == '__main__':
    # create_train_test_split_skeleton()
    # get_bounding_box_all_videos()
    # extract_and_crop_all_videos()
    # recalculate_bounding_box_missing_videos()
    recrop_frames_missing_videos()
