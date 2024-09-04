import os
import json
import cv2

set_ ='test'
grounding_anno_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2' + set_ +'.json'
folder_path = '/data/zsheng/Data_T5_ViteVQA/data/grounding_data/t1s2'+ set_

with open(grounding_anno_path, 'r') as file:
    anno = json.load(file)

temporal_span_list = []
excel_list = [['Question ID', 'Box Number']]
for i, element in enumerate(anno['data']):
    question_id = element['question_id']
    video_id = element['video_id']
    spatial_temporal_gt = element['spatial_temporal_gt']

    # if len(spatial_temporal_gt) >= 3:
    #     print(question_id, video_id)

    temporal_span_list.append(len(spatial_temporal_gt))
    excel_list.append([question_id, len(spatial_temporal_gt)])

import statistics
sum_temporal_span = sum(temporal_span_list)
average_temporal_span = statistics.mean(temporal_span_list)
median_temporal_span = statistics.median(temporal_span_list)

print(f"{set_} set, ann span sum: {sum_temporal_span}, ave: {average_temporal_span}, med:{median_temporal_span}")


import openpyxl

def write_list_to_excel(data, filename):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    for row in data:
        sheet.append(row)

    workbook.save(filename)


write_list_to_excel(excel_list, 'Distribution_of_temporal_span_number_on_'+ set_ +'.xlsx')