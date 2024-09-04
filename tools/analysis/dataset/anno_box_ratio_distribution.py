import os
import json
import cv2

set_ ='val'

grounding_anno_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2' + set_ +'.json'
video_folder_path = '/data/zsheng/Data_T5_ViteVQA/data/fps10_video'

with open(grounding_anno_path, 'r') as file:
    anno = json.load(file)

box_list = []
excel_list = [['Question ID', 'Box Ratio']]
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
    for span in spatial_temporal_gt:
      temporal_gt = span['temporal_gt']
      star_id, end_id = int(temporal_gt[0])*10, int(temporal_gt[1])*10

      ground_id_list = list(range(star_id, end_id + 1))

      ground_frames += (end_id - star_id + 1)

      box_list.append(ground_frames)
      ratio = ground_frames / frames_length
      excel_list.append([question_id, ratio])



import openpyxl

def write_list_to_excel(data, filename):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    for row in data:
        sheet.append(row)

    workbook.save(filename)


# 将列表写入Excel文件
write_list_to_excel(excel_list, 'Distribution_of_Box_Ratio_on_'+ set_ +'.xlsx')