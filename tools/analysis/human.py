import cv2
import os
import json
import numpy as np
import openpyxl

def calculate_iou(box1, box2):
    """
    Intersection over Union (IoU)
    box1, box2: (x1, y1, x2, y2), (x1, y1): top-left coordinates, (x2, y2): bottom right coordinate.
    return: IoU
    """
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    x1_inter = max(x1_box1, x1_box2)
    y1_inter = max(y1_box1, y1_box2)
    x2_inter = min(x2_box1, x2_box2)
    y2_inter = min(y2_box1, y2_box2)

    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    area_box1 = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    area_box2 = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)
    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area

    return iou

def check_iou(gt_bbox, pred_bbox, threshold = None):
  flag = False
  bbox_iou = -1
  max_iou = 0
  # for _,pred_bbox in enumerate(pred_bboxs):
  assert pred_bbox[0]<=pred_bbox[2] and pred_bbox[1]<=pred_bbox[3]
  bbox_iou = calculate_iou(gt_bbox, pred_bbox)
  if bbox_iou > max_iou:
      max_iou = bbox_iou

  if max_iou > threshold:
      flag = True        
      
  return flag

import editdistance
def get_anls(s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= .5 else 0.
        return anls



if __name__ == "__main__":
    set_ = 'test'
    sample100_path = '/data/zsheng/Data_T5_ViteVQA/tools/human_base_study/100_sample_on_test.xlsx'
    sample100_json_path = '/data/zsheng/Data_T5_ViteVQA/tools/human_base_study/human_study_anno_100_samples_anno.json'
    grounding_anno_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2'+set_+'.json'
    qa_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/qa_sub_t1s2'+set_+'.json'
    ocr_info_path = '/data/zsheng/Data_T5_ViteVQA/data/fps10_ocr_detection'
    ocr_file_path = os.path.join(ocr_info_path, set_)
    vocab_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/vocabulary/fixed_vocab_top5k_t1s2.txt'


    workbook = openpyxl.load_workbook(sample100_path)
    sheet = workbook['Sheet1']
    sample_list = []
    for row in sheet.iter_rows(values_only=True):
        row_list = list(row)
        sample_list.append({'question_id':row_list[0], 'video_id':row_list[1], 'human_answer':row_list[2]})

    # print(sample_list)

    with open(sample100_json_path,'r') as file:
          human_anno = json.load(file)

    with open(grounding_anno_path,'r') as file:
          grounding_anno = json.load(file)

    with open(qa_path,'r') as file:
        qa_file = json.load(file)

    # obtain vocabulary
    with open(vocab_path, 'r') as f:
        vocab_file = f.readlines()
    vocab_file = [line.rstrip('\n') for line in vocab_file]
    vocab_list = list(vocab_file)


    # 100 samples
    total_qa_acc, total_anls_acc, total_iou_acc, total_gqa_acc = [], [], [], []
    for sample in sample_list:
        
        
        human_qa, human_iou, human_gqa = 0, 0, 0

        v_id = sample['video_id']
        q_id = sample['question_id']
        video_path = os.path.join(ocr_file_path, v_id+'.npy')

        # if q_id != 2767: continue

        # obtain GT answer 
        for i,qa in enumerate(qa_file['data']):
            if qa['question_id'] == q_id:
                gt_ans = qa['answers']

        # obtain GT box
        for i, gt_element in enumerate(grounding_anno['data']):
            if gt_element['question_id'] == q_id:
                gt_info = gt_element['spatial_temporal_gt']
                for j, span in enumerate(gt_info):
                    gt_frame = span['temporal_gt']
                    gt_box = span['bbox_gt']

        # obtain human answer
        human_ans = sample['human_answer']

        # obtain human grounding results
        for i, qa in enumerate(human_anno['data']):
              if qa['question_id'] == q_id:
                  span = qa['spatial_temporal_human'][0]
                  human_frame = span['temporal_human']
                #   human_frame_st = int(human_frame[0]*10)+1
                #   human_frame_ed = int(human_frame[1]*10)+1
                  human_box = span['bbox_human']
        


        # QA
        for ans in gt_ans:
            if str(human_ans).lower() == ans.lower():
                human_qa = 1
        if human_qa == 0:
            print(f"human_qa=0, q id:{sample['question_id']}")
            print(f"human_ans:{str(human_ans).lower()}, gt_ans:{ans.lower()}, error!")


        # ANLS
        anls_score = 0
        for ans in gt_ans:
          score = get_anls(str(human_ans).lower(), ans.lower())
          if score>anls_score:
              anls_score = score
        # if human_qa == 0:
        print(f"anls_score:{anls_score}")
  
          

      # IoU
        pred_frame = list(human_box.keys())[1]
        gt_frame = list(gt_box.keys())
        if pred_frame in gt_frame:
            pred_box = human_box[pred_frame]
            gt_box = gt_box[pred_frame]
            assert pred_box[0]<=pred_box[2] and pred_box[1]<=pred_box[3]
            result = check_iou(pred_box, gt_box, threshold=0.5) 
            if result:
                human_iou = 1
            else:
                human_iou = 0
        else:
            human_iou = 0
        
        # if human_iou == 0:
        # print(f"iou=0, q id:{sample['question_id']}")
        # print(f"human_ans:{str(human_ans).lower()}")
        print(f"gt_frame:{gt_frame}, pred_frame:{human_box.keys()}")

        # GQA
        if human_qa and human_iou:
            human_gqa = 1
        else:
            human_gqa = 0
        
        if human_qa == 0 and human_iou == 0:
            print(f"iou=0, q id:{sample['question_id']}")
            print(f"gt_frame:{gt_frame}, pred_frame:{human_box.keys()}")
            print(f"human_ans:{str(human_ans).lower()}, gt_ans:{ans.lower()}, error!")

        total_qa_acc.append(human_qa)
        total_anls_acc.append(anls_score)
        total_iou_acc.append(human_iou)
        total_gqa_acc.append(human_gqa)
    
    import statistics
    qa_acc = round(statistics.mean(total_qa_acc), 4)
    anls_acc = round(statistics.mean(total_anls_acc), 4)
    iou_acc = round(statistics.mean(total_iou_acc), 4)
    gqa_acc = round(statistics.mean(total_gqa_acc), 4)
    print(f"{set_} set: qa_acc={qa_acc}, qa_anls={anls_acc}, iou_acc={iou_acc}, gqa_acc={gqa_acc}")