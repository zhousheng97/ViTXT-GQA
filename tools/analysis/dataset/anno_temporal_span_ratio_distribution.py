import os
import json
import cv2

set_ ='val'
grounding_anno_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2' + set_ +'.json'
video_folder_path = '/data/zsheng/Data_T5_ViteVQA/data/fps10_video'

with open(grounding_anno_path, 'r') as file:
    anno = json.load(file)

ratio_list=[]
excel_list = [['Question ID', 'Segment Ratio']]
for i, element in enumerate(anno['data']):
    question_id = element['question_id']
    video_id = element['video_id']
    spatial_temporal_gt = element['spatial_temporal_gt']
    ground_frames = 0
    
    video_path = os.path.join(video_folder_path, video_id+'.mp4')
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    all_seg_span = 0
    for span in spatial_temporal_gt:
        temporal_gt = span['temporal_gt']
        span_len = temporal_gt[1] - temporal_gt[0]
        all_seg_span += span_len

        seg_ratio = all_seg_span / element['duration']
        excel_list.append([question_id, seg_ratio])
        ratio_list.append(seg_ratio)

import statistics
average_ratio = statistics.mean(ratio_list)
median_ratio = statistics.median(ratio_list)

print(f"{set_} set, segment ratio in the whole video: average: {average_ratio}, median:{median_ratio}")


import openpyxl

def write_list_to_excel(data, filename):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    for row in data:
        sheet.append(row)

    workbook.save(filename)


write_list_to_excel(excel_list, 'Distribution_of_Temporal_Span_Ratio_on_'+ set_ +'.xlsx')