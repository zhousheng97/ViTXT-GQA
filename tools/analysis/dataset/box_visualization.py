import cv2
import os
import numpy as np
import json
from PIL import Image, ImageDraw

set_ = 'val'
frame_folder = '/data/zsheng/T5-ViteVQA/data/fps10_frames'
ocr_detection_folder = '/data/zsheng/T5-ViteVQA/data/fps10_ocr_detection/' + set_
visualize_folder = '/data/zsheng/T5-ViteVQA/tools/ground_process/dataset_analyze/visualize'

grounding_anno = '/data/zsheng/T5-ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2'+ set_ +'.json'
with open(grounding_anno,'r') as file:
  grounding_anno = json.load(file)

for _,q_element in enumerate(grounding_anno['data']):
  q_id = q_element['question_id']
  v_id = q_element['video_id']
  spatial_temporal_gt = q_element['spatial_temporal_gt']

  save_q_path = os.path.join(visualize_folder,str(q_id))
  save_v_path = os.path.join(save_q_path,v_id)

  for span in spatial_temporal_gt:
      bbox_gt = span['bbox_gt']
      for f in bbox_gt:
          frame_id = int(f)+1
          x1, y1, x2, y2 = bbox_gt[f]
          x2, y2 = x1+w, y1+h

          
          image_path = os.path.join(frame_folder, v_id, str(frame_id)+'.jpg')
          save_path = os.path.join(save_v_path,str(frame_id)+'.jpg')
          if os.path.exists(save_path):
              continue

          if os.path.exists(image_path):
              image = Image.open(image_path)
          else:
              print(f"image not exist:{image_path}")
              video_path = os.path.join(os.path.join(frame_folder, v_id))
              frame_id = len(os.listdir(video_path))
              image = Image.open(os.path.join(frame_folder, v_id, str(frame_id-1)+'.jpg'))  
          draw = ImageDraw.Draw(image)
          draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline='red', width=3)

          if not os.path.exists(save_q_path):
              os.makedirs(save_q_path)
          if not os.path.exists(save_v_path):
              os.makedirs(save_v_path)
          image.save(save_path)


  if os.path.exists(save_v_path):
      images = sorted(os.listdir(save_v_path))
  
  for frame in images:
      frame_id = frame.split('.')[0]
      image = Image.open(os.path.join(save_v_path, frame))
      ocr_info_path = os.path.join(ocr_detection_folder, v_id+'.npy')
      ocr_info = np.load(ocr_info_path, allow_pickle=True).item()
      
      if len(ocr_info)>int(frame_id):
              ocr_list = ocr_info[frame_id]
      else:
          print(f"frame: {frame_id} not exist in ocr_info")
          ocr_list = ocr_info[str(len(ocr_info))]  #

      for ocr in ocr_list:
          points = ocr['points']
          x1 = min(points[0],points[6])
          y1 = min(points[1],points[3])
          x2 = max(points[2],points[4])
          y2 = max(points[5],points[7])
          point = [int(x1), int(y1), int(x2), int(y2)]

          draw = ImageDraw.Draw(image)
          draw.rectangle(point, outline='blue', width=3)

      image.save(os.path.join(save_v_path, frame))



  # draw = ImageDraw.Draw(image)
  # draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline='red', width=2)

  # if not os.path.exists(save_q_path):
  #     os.makedirs(save_q_path)
  # if not os.path.exists(save_v_path):
  #     os.makedirs(save_v_path)
  # image.save(save_path)

