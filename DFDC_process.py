import os
import cv2
import json
import threading
import numpy as np
from tqdm import tqdm
from pathlib import Path
from deepface import DeepFace

KPS = 0  # if 0 or negative KPS=FPS i.e. skip NO frames
KPS_string = str(KPS) if KPS > 0 else 'all'
IMAGE_EXTENSION = '.jpg'  # .png takes way more space
FACE_RESOLUTION = 224

RUN_HOME = Path(__file__).parent.resolve()
VIDEO_HOME = os.path.join(RUN_HOME, 'dfdc_preview_set')
FRAMES_HOME = os.path.join(RUN_HOME, f'dfdc_preview-frames-KPS-{KPS_string}')
ALIGNED_FRAMES_HOME = os.path.join(RUN_HOME, f'dfdc_preview-aligned-{FACE_RESOLUTION}-KPS-{KPS_string}')

N_THREADS = 5
Path(FRAMES_HOME).mkdir(parents=True, exist_ok=True)
Path(ALIGNED_FRAMES_HOME).mkdir(parents=True, exist_ok=True)


def process_subset_videos(vid_set_list):
    
    for vid_path_key, train_or_test in tqdm(vid_set_list):
        if not vid_path_key.endswith('.mp4'): continue

        video_path_full = os.path.join(VIDEO_HOME, vid_path_key)
        train_or_test = data[vid_path_key]["set"]

        out_frames_dir = os.path.join(FRAMES_HOME, train_or_test, vid_path_key[:-4])
        out_aligned_dir = os.path.join(ALIGNED_FRAMES_HOME, train_or_test, vid_path_key[:-4])
        
        # print(out_frames_dir)
        Path(out_frames_dir).mkdir(parents=True, exist_ok=True)
        Path(out_aligned_dir).mkdir(parents=True, exist_ok=True)
        
        vidcap = cv2.VideoCapture(video_path_full)
        success, image = vidcap.read()
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        KPS_divider = fps if KPS <= 0 else KPS
        count, hop = 0, round(fps//KPS_divider)  # hop = round(fps/kps) where kps is keyframes per sec.

        while success:
            if count % hop == 0:
                frame_name = f'{count:05d}.{IMAGE_EXTENSION}'
                # frame_filepath = os.path.join(out_frames_dir, frame_name)
                # if not os.path.exists(frame_filepath):
                #     cv2.imwrite(frame_filepath, image)
                
                # Detect, Align, Crop, save face
                aligned_face_image_path = os.path.join(out_aligned_dir, frame_name)
                try:
                    aligned_face_image = DeepFace.detectFace(img_path=image, 
                                                            target_size=(FACE_RESOLUTION, FACE_RESOLUTION), 
                                                            detector_backend='retinaface',
                                                            enforce_detection=True)
                    cv2.imwrite(aligned_face_image_path, aligned_face_image[:, :, ::-1]*255)
                except ValueError:
                    with open(f"./dfdc_errors-{FACE_RESOLUTION}-KPS-{KPS_string}.txt", "a") as f:
                        f.write(out_aligned_dir+'\n')
            
            count += 1
            success, image = vidcap.read()
        vidcap.release()


if __name__ == '__main__':
    with open(os.path.join(VIDEO_HOME, 'dataset.json'), 'r') as f:
        data = json.load(f)

        full_vid_set_list = []
        for vid_path_key in data:
            full_vid_set_list.append((vid_path_key, data[vid_path_key]["set"]))
        
        chunk_size = len(full_vid_set_list)//N_THREADS
        chunks = [full_vid_set_list[i:i + chunk_size] for i in range(0, len(full_vid_set_list), chunk_size)]
        
        threads = []
        for chunk in chunks:
            # print(chunk)
            # exit(0)
            process_subset_videos(chunk)
            # t = threading.Thread(target=process_subset_videos, args=(chunk,))
            # threads.append(t)
            # t.start()

        for _t in threads:
            _t.join()