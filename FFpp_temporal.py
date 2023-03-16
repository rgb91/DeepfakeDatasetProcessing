# pylint: disable=line-too-long invalid-name missing-module-docstring

import os
import csv
import shutil
import random
import warnings
import pandas as pd
from pathlib import Path

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

videos = ['002_006.mp4', '004_982.mp4', '005_010.mp4', '007_132.mp4', '016_209.mp4', '024_073.mp4', '042_084.mp4',
          '045_889.mp4', '053_095.mp4', '057_070.mp4', '066_062.mp4', '085_124.mp4', '088_060.mp4', '089_065.mp4',
          '092_098.mp4', '100_077.mp4', '101_096.mp4', '112_892.mp4', '124_085.mp4', '125_038.mp4', '131_518.mp4',
          '132_007.mp4', '151_225.mp4', '165_137.mp4', '166_167.mp4', '182_242.mp4', '184_205.mp4', '196_310.mp4',
          '198_106.mp4', '201_203.mp4', '204_230.mp4', '209_016.mp4', '225_151.mp4', '238_282.mp4', '239_218.mp4',
          '271_264.mp4', '286_267.mp4', '297_270.mp4', '339_392.mp4', '344_020.mp4', '348_202.mp4', '351_346.mp4',
          '387_311.mp4', '392_339.mp4', '393_405.mp4', '397_602.mp4', '412_274.mp4', '428_466.mp4', '438_434.mp4',
          '439_441.mp4', '441_439.mp4', '463_464.mp4', '471_455.mp4', '484_415.mp4', '516_555.mp4', '521_517.mp4',
          '526_436.mp4', '528_510.mp4', '532_544.mp4', '566_617.mp4', '569_921.mp4', '584_823.mp4', '604_703.mp4',
          '611_760.mp4', '623_630.mp4', '632_548.mp4', '640_638.mp4', '649_816.mp4', '653_601.mp4', '715_721.mp4',
          '717_684.mp4', '731_741.mp4', '744_674.mp4', '748_355.mp4', '752_751.mp4', '763_930.mp4', '782_787.mp4',
          '792_903.mp4', '793_768.mp4', '809_799.mp4', '810_838.mp4', '814_871.mp4', '829_808.mp4', '831_508.mp4',
          '870_001.mp4', '872_873.mp4', '877_886.mp4', '878_866.mp4', '894_848.mp4', '899_914.mp4', '907_795.mp4',
          '915_895.mp4', '916_783.mp4', '918_934.mp4', '927_912.mp4', '952_882.mp4', '957_959.mp4', '969_897.mp4',
          '996_056.mp4', '999_960.mp4']

# frames_source_path = r'/mnt/d/DATASETS/FFpp-c23-frames-aligned-224/manipulated_sequences'
# frames_destination_path = r'/mnt/d/DATASETS/FFpp-mixed-segments-kim-ngan/faces_unspliced/'

# datasets = ['Deepfakes', 'FaceShifter', 'FaceSwap']
# for d in datasets:
#     for v in videos:
#         vid_name = v.split('.')[0]
#         source_path = os.path.join(frames_source_path, d, vid_name)
#         destination_path = os.path.join(frames_destination_path, d, vid_name)        
#         Path(destination_path).mkdir(parents=True, exist_ok=True)

#         shutil.copytree(source_path, destination_path, dirs_exist_ok=True)

# real_path = r'./FFpp-c23-224/original_sequences/youtube'
# fake_path = r'./FFpp-mixed-segments-kim-ngan/faces_unspliced/'
# datasets = ['Deepfakes', 'FaceShifter', 'FaceSwap']
datasets = ['fake_DF', 'fake_FSh', 'fake_NT', 'fake_FS', 'fake_F2F']  # 'fake_F2F'
dataset_root = os.path.join('..', 'FFpp-c23-224')
dataset_config = os.path.join('..', 'FFpp-c23-224-config')
dest_root_1 = os.path.join('..', 'FFpp_temporal_one_segment')
dest_root_2 = os.path.join('..', 'FFpp_temporal_two_segments')


def get_train_or_test(video_name):
    if len(video_name.split('_')) == 1:  # 'real'
        test_real = pd.read_csv(os.path.join(dataset_config, 'test_real.csv'), header=None)[0].values.tolist()
        if int(video_name) in test_real:
            return 'test'
        return 'train'
    else:  # 'fake'
        test_fake = pd.read_csv(os.path.join(dataset_config, 'test_fake.csv'), header=None)[0].values.tolist()
        if video_name in test_fake:
            return 'test'
        return 'train'


def scratch_calculations():
    """
    Create the ground truth for temporal dataset with two (2) fake segments
    :return: None
    """

    df = pd.read_csv(r'dumped/temporal_ground_truth_1.csv')

    count = 0
    for video_name in videos:
        rows = df.loc[df['id'] == video_name]
        if rows.empty:
            print('Row is empty!', video_name)
            continue
        if len(rows) > 1:
            print('Multiple rows:', video_name)
            continue

        # fake_start, fake_end = int(rows['1st-fake']), int(rows['last-fake'])
        total = int(rows['total'])

        train_or_test = get_train_or_test(video_name=video_name)
        real_frames_dir = os.path.join(dataset_root, train_or_test, 'real', video_name.split('_')[0])
        # fake_frames_dir = os.path.join(dataset_root, train_or_test, 'real', video_name.split('_')[0])

        if len(os.listdir(real_frames_dir)) != total:
            print('Number of frames should be equal to total: ', video_name)
            continue

        if total < 400:
            count += 1
    print('Total less than 400: ', count)


def find_minimum_total_frames(video_path):
    min_len = len(os.listdir(video_path))
    for d in ['fake_DF', 'fake_FSh', 'fake_NT', 'fake_FS']:
        video_path_new = video_path.replace('fake_F2F', d)
        min_len = min(len(os.listdir(video_path_new)), min_len)

    return min_len


def calculate_total_frames():
    path_1 = os.path.join(dataset_root, 'train', 'fake_F2F')
    path_2 = os.path.join(dataset_root, 'test', 'fake_F2F')

    videos_1 = [os.path.join(dataset_root, 'train', 'fake_F2F', vid) for vid in os.listdir(path_1)]
    videos_2 = [os.path.join(dataset_root, 'test', 'fake_F2F', vid) for vid in os.listdir(path_2)]
    all_videos = videos_1 + videos_2

    df = pd.DataFrame(columns=['video_name', 'total_frames'])
    for video_path in all_videos:
        # total_frames = len(os.listdir(video_path))
        total_frames = find_minimum_total_frames(video_path)
        df = df.append({'video_name': os.path.basename(video_path),
                        'total_frames': total_frames}, ignore_index=True)

    df.to_csv('fake_video_lengths.csv', index=False)


def select_100_videos():
    in_df = pd.read_csv('fake_video_lengths.csv')
    shortlist = in_df.loc[in_df['total_frames'] >= 500]
    final_100 = shortlist.sample(n=100)
    final_100.to_csv('temporal_videos_100.csv', index=False)


def select_fake_segments():
    out_df_1 = pd.DataFrame(columns=['video_name', 'total_frames', 'fake_start', 'fake_end'])
    out_df_2 = pd.DataFrame(
        columns=['video_name', 'total_frames', 'fake1_start', 'fake1_end', 'fake2_start', 'fake2_end'])

    in_df = pd.read_csv('temporal_videos_100.csv')
    # videos = [os.path.basename(vid) for vid in in_df['video_path']]
    videos = in_df['video_name'].values.tolist()
    videos_lengths = [int(tot_len) for tot_len in in_df['total_frames']]

    for v_name, v_len in zip(videos, videos_lengths):
        # process One-fake-segment
        fake_start = random.randint(125, v_len // 2)  # fakes starts from somewhere between 5 seconds to half of video
        fake_len = random.choice([125, 150, 175])  # length of fake segment is 5, 6, or 7 seconds

        out_df_1 = out_df_1.append({
            'video_name': v_name,
            'total_frames': v_len,
            'fake_start': int(fake_start),
            'fake_end': int(fake_start + fake_len),
        }, ignore_index=True)

        # process Two-fake-segments
        fake1_start = random.randint(50, 125)  # fakes starts from somewhere between 2 seconds to 5 seconds
        fake1_len = random.choice([120, 125])  # length of fake segment is 4 or 5 seconds
        fake2_start = random.randint(v_len // 2, (v_len // 2) + 75)  # fakes starts in second half of video
        fake2_len = random.choice([125, 150])  # length of fake segment is 5 or 6 seconds

        out_df_2 = out_df_2.append({
            'video_name': v_name,
            'total_frames': v_len,
            'fake1_start': int(fake1_start),
            'fake1_end': int(fake1_start + fake1_len),
            'fake2_start': int(fake2_start),
            'fake2_end': int(fake2_start + fake2_len),
        }, ignore_index=True)
    out_df_1.to_csv('temporal_one_segment.csv', index=False)
    out_df_2.to_csv('temporal_two_segments.csv', index=False)


def make_dataset_one_segment():
    df = pd.read_csv('temporal_one_segment.csv')
    data = df.values.tolist()
    debug_list = []
    for d in datasets:
        d_err_count = 0
        for row in tqdm(data, desc=d):
            video_name = row[0]
            video_len = int(row[1])
            fake_start, fake_end = int(row[2]), int(row[3])

            which_set = get_train_or_test(video_name)
            src_fake_dir = os.path.join(dataset_root, which_set, d, video_name)
            src_real_0_dir = os.path.join(dataset_root, get_train_or_test(video_name.split('_')[0]), 'real',
                                          video_name.split('_')[0])
            src_real_1_dir = os.path.join(dataset_root, get_train_or_test(video_name.split('_')[1]), 'real',
                                          video_name.split('_')[1])
            dest_dir = os.path.join(dest_root_1, d, video_name)

            if len(os.listdir(src_fake_dir)) == len(os.listdir(src_real_0_dir)):
                src_real_dir = src_real_0_dir
            elif len(os.listdir(src_fake_dir)) == len(os.listdir(src_real_1_dir)):
                src_real_dir = src_real_1_dir
            else:
                d_err_count += 1
                debug_list.append([d, video_name, len(os.listdir(src_fake_dir)), len(os.listdir(src_real_0_dir)),
                                   len(os.listdir(src_real_1_dir))])
                continue

            fake_frames = [os.path.join(src_fake_dir, f_name) for f_name in os.listdir(src_fake_dir)]
            real_frames = [os.path.join(src_real_dir, f_name) for f_name in os.listdir(src_real_dir)]
            final_frames = real_frames[:fake_start] + fake_frames[fake_start:fake_end] + real_frames[fake_end:]
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            for f_path in final_frames:
                shutil.copyfile(f_path, os.path.join(dest_dir, os.path.basename(f_path)))
            # print(d, video_name)
            # exit(1)

        print(d, d_err_count)
    with open('debug_list_3.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(['dataset', 'video', 'fake', 'real_0', 'real_1'])
        write.writerows(debug_list)
    print('END')
    pass


def make_dataset_two_segments():
    df = pd.read_csv('temporal_two_segments.csv')
    data = df.values.tolist()
    debug_list = []
    for d in datasets:
        d_err_count = 0
        for row in tqdm(data, desc=d):
            video_name = row[0]
            video_len = int(row[1])
            fake1_start, fake1_end = int(row[2]), int(row[3])
            fake2_start, fake2_end = int(row[4]), int(row[5])

            which_set = get_train_or_test(video_name)
            src_fake_dir = os.path.join(dataset_root, which_set, d, video_name)
            src_real_0_dir = os.path.join(dataset_root, get_train_or_test(video_name.split('_')[0]), 'real',
                                          video_name.split('_')[0])
            src_real_1_dir = os.path.join(dataset_root, get_train_or_test(video_name.split('_')[1]), 'real',
                                          video_name.split('_')[1])
            dest_dir = os.path.join(dest_root_2, d, video_name)

            if len(os.listdir(src_fake_dir)) == len(os.listdir(src_real_0_dir)):
                src_real_dir = src_real_0_dir
            elif len(os.listdir(src_fake_dir)) == len(os.listdir(src_real_1_dir)):
                src_real_dir = src_real_1_dir
            else:
                d_err_count += 1
                debug_list.append([d, video_name, len(os.listdir(src_fake_dir)), len(os.listdir(src_real_0_dir)),
                                   len(os.listdir(src_real_1_dir))])
                continue

            fake_frames = [os.path.join(src_fake_dir, f_name) for f_name in os.listdir(src_fake_dir)]
            real_frames = [os.path.join(src_real_dir, f_name) for f_name in os.listdir(src_real_dir)]
            final_frames = real_frames[:fake1_start] + \
                           fake_frames[fake1_start:fake1_end] + \
                           real_frames[fake1_end:fake2_start] + \
                           fake_frames[fake2_start:fake2_end] + \
                           real_frames[fake2_end:]
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            for f_path in final_frames:
                shutil.copyfile(f_path, os.path.join(dest_dir, os.path.basename(f_path)))

        print(d, d_err_count)
    with open('debug_list_3.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(['dataset', 'video', 'fake', 'real_0', 'real_1'])
        write.writerows(debug_list)
    print('END')
    pass


if __name__ == '__main__':
    # calculate_total_frames()
    # select_100_videos()
    # select_fake_segments()
    make_dataset_two_segments()

# for d in datasets:
#     for vid in os.listdir(os.path.join(fake_path, d)):
#         vid_mp4 = vid + '.mp4'
#         rows = df.loc[df['id'] == vid_mp4]
#         if rows.empty:
#             print('Row is empty!', vid)
#             continue
#         if len(rows) > 1:
#             print(vid)
#             continue
#
#         fake_start, fake_end = int(rows['1st-fake']), int(rows['last-fake'])
#
#         real_vid_dir = os.path.join(real_path, vid.split('_')[0])
#         fake_vid_dir = os.path.join(fake_path, d, vid)
#
#         real_frames = os.listdir(real_vid_dir)
#         fake_frames = os.listdir(fake_vid_dir)
#
#         # assert len(real_frames) == len(fake_frames), "Both directory should have same length."
#         if len(real_frames) != len(fake_frames):
#             print(real_vid_dir)
#             print(fake_vid_dir)
#             print(len(real_frames), len(fake_frames))
#             print()
#         # exit(1)
