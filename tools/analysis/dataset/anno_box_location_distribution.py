import os
import json
import cv2

def calculate_bbox_position(bbox, image_size):
    x1, y1, x2, y2 = bbox   
    img_width, img_height = image_size   

    bbox_center_x = (x1 + x2) / 2   
    bbox_center_y = (y1 + y2) / 2   

    if bbox_center_x < img_width / 2:
        if bbox_center_y < img_height / 2:
            return "top left" 
        else:
            return "bottom left"
    else:
        if bbox_center_y < img_height / 2:
            return "top right"
        else:
            return "bottom right"


set_ ='test'
# 遍历每个文件夹
grounding_anno_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2' + set_ +'.json'
folder_path = '/data/zsheng/Data_T5_ViteVQA/data/grounding_data/t1s2'+ set_

with open(grounding_anno_path, 'r') as file:
    anno = json.load(file)

box_list = []
excel_list = [['Question ID', 'Box Number']]
for i, element in enumerate(anno['data']):
    question_id = element['question_id']
    video_id = element['video_id']
    spatial_temporal_gt = element['spatial_temporal_gt']
    ground_frames = 0
    ground_ids_list = []
    for span in spatial_temporal_gt:
      temporal_gt = span['temporal_gt']
      star_id, end_id = int(temporal_gt[0])*10, int(temporal_gt[1])*10
      ground_id_list = list(range(star_id, end_id + 1))
      ground_frames += (end_id - star_id + 1)
      # Top left, bottom left, top right, bottom right, center
      height, width = element['height'], element['width']
      for span in spatial_temporal_gt:
          temporal_gt = span['temporal_gt']
          star_id, end_id = int(temporal_gt[0])*10, int(temporal_gt[1])*10
          ground_id_list = list(range(star_id, end_id + 1))
          ground_frames += (end_id - star_id + 1)
          box_gt = span['bbox_gt']
          for _, box in enumerate(box_gt): 
              x, y, area_ratio = 0, 0, 0
              box = box_gt[box]
              position = calculate_bbox_position(box, (width, height))
              excel_list.append([question_id, position])


import openpyxl

def write_list_to_excel(data, filename):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    for row in data:
        sheet.append(row)

    workbook.save(filename)


write_list_to_excel(excel_list, 'Distribution_of_Box_Location_on_'+ set_ +'.xlsx')