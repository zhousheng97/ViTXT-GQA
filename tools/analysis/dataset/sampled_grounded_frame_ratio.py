import os
import json
import cv2

set_ ='val'
# 遍历每个文件夹
grounding_anno_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2' + set_ +'.json'
folder_path = '/data/zsheng/Data_T5_ViteVQA/data/grounding_data/t1s2'+ set_

# 定义视频帧采样数
sample_size = 10000


with open(grounding_anno_path, 'r') as file:
    anno = json.load(file)

average_annotated_sampled_frames_percentage, average_total_frames_percentage = [], []


for i, element in enumerate(anno['data']):
    question_id = element['question_id']
    video_id = element['video_id']
    spatial_temporal_gt = element['spatial_temporal_gt']

    video_path = os.path.join(folder_path, str(video_id), str(video_id) + '.mp4')
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

  
    ground_frames = 0
    ground_ids_list = []
    for span in spatial_temporal_gt:
        temporal_gt = span['temporal_gt']
        star_id, end_id = int(temporal_gt[0])*10, int(temporal_gt[1])*10
        ground_id_list = list(range(star_id, end_id + 1))
        ground_frames += (end_id - star_id + 1)

        if len(spatial_temporal_gt)>1:
            ground_ids_list.extend(ground_id_list)
        else:
            ground_ids_list = ground_id_list
            continue


    sample_id_list = []
    step = frames_length // sample_size

    if frames_length <= sample_size:
        sample_id_list = list(range(frames_length))
    else:
        lst = list(range(frames_length))
        step = frames_length // sample_size  
        sample_id_list = [lst[i * step] for i in range(sample_size)]


    annotated_total_frames_percentage = (ground_frames / frames_length) * 100
    average_total_frames_percentage.append(annotated_total_frames_percentage)


    set1 = set(ground_ids_list)
    set2 = set(sample_id_list)
    intersection = set1.intersection(set2)
    annotated_sampled_frames_percentage = (len(intersection) / len(set1)) * 100
    average_annotated_sampled_frames_percentage.append(annotated_sampled_frames_percentage)
    if annotated_sampled_frames_percentage <= 10:
        continue


    print(f"q_id: {question_id}")
    print(f"total anno frame ratio: {annotated_total_frames_percentage}%")
    print(f"sampled anno frame ratio: {annotated_sampled_frames_percentage}%")
    print()
import statistics
average_frames_percentage = sum(average_total_frames_percentage) / len(average_total_frames_percentage)
median_frames_percentage = statistics.median(average_total_frames_percentage)
print(f"{set_} set,  ave:{average_frames_percentage}, med:{median_frames_percentage}")
average_annotated_sampled_frames_percentage = sum(average_annotated_sampled_frames_percentage) / len(average_annotated_sampled_frames_percentage)
print(f"{set_} set, uniform sampled {sample_size} frame, Average ratio of anno frames in sampled frames to total anno frames: {average_annotated_sampled_frames_percentage}")

