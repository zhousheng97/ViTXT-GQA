'''统计整个数据集的标注的box的平均数和中位数'''

import os
import json
import cv2

set_ ='test'
grounding_anno_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2' + set_ +'.json'

with open(grounding_anno_path, 'r') as file:
    anno = json.load(file)

box_area_list = []
excel_list = [['Question ID', 'Box Number']]
for i, element in enumerate(anno['data']):
    question_id = element['question_id']
    video_id = element['video_id']
    spatial_temporal_gt = element['spatial_temporal_gt']
    ground_frames = 0
    ground_ids_list = []
    height, width = element['height'], element['width']
    for span in spatial_temporal_gt:
      temporal_gt = span['temporal_gt']
      star_id, end_id = int(temporal_gt[0])*10, int(temporal_gt[1])*10
   
      ground_id_list = list(range(star_id, end_id + 1))

      ground_frames += (end_id - star_id + 1)
      box_gt = span['bbox_gt']
  
      for _, box in enumerate(box_gt): 
          x, y, area_ratio = 0, 0, 0
          x1, y1, x2, y2 = box_gt[box]

          area_ratio = (abs(x1 - x2)*abs(y1 - y2)) / (height*width)
          excel_list.append([question_id, area_ratio])

import openpyxl

def write_list_to_excel(data, filename):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    for row in data:
        sheet.append(row)

    workbook.save(filename)


write_list_to_excel(excel_list, 'Distribution_of_Box_Area_Proportion_'+ set_ +'.xlsx')