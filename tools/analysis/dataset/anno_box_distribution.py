import os
import json
import cv2

set_ ='test'
grounding_anno_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2' + set_ +'.json'
folder_path = '/data/zsheng/Data_T5_ViteVQA/data/grounding_data/t1s2'+ set_

with open(grounding_anno_path, 'r') as file:
    anno = json.load(file)

box_list = []
box_sum = 0
excel_list = [['Question ID', 'Box Number']]
for i, element in enumerate(anno['data']):
    question_id = element['question_id']
    video_id = element['video_id']
    spatial_temporal_gt = element['spatial_temporal_gt']

    ground_boxs = 0
    for span in spatial_temporal_gt:
        temporal_gt = span['temporal_gt']
        star_id, end_id = int(temporal_gt[0])*10, int(temporal_gt[1])*10
        box_num = len(span['bbox_gt'])
        box_sum += box_num
        ground_boxs += box_num

    
    box_list.append(ground_boxs)
    excel_list.append([question_id, ground_boxs])


import statistics
sum_box = sum(box_list)
average_box = statistics.mean(box_list)
median_box = statistics.median(box_list)

print(f"{set_} set, box sum: {sum_box}, ave: {average_box}, med:{median_box}")

import openpyxl

def write_list_to_excel(data, filename):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    for row in data:
        sheet.append(row)

    workbook.save(filename)


write_list_to_excel(excel_list, 'Distribution_of_Box_number_on_'+ set_ +'.xlsx')