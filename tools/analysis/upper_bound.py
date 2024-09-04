import os
import numpy as np
import json
import editdistance

def get_anls(s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= .5 else 0.
        return anls


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


if __name__ == "__main__":
    set_ = 'test'

    grounding_anno_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2'+set_+'.json'
    qa_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/qa_sub_t1s2'+set_+'.json'
    ocr_info_path = '/data/zsheng/Data_T5_ViteVQA/data/fps10_ocr_detection_ClipOCR' #   '/data/zsheng/Data_T5_ViteVQA/data/fps10_ocr_detection'  
    ocr_file_path = os.path.join(ocr_info_path, set_)
    vocab_path = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/vocabulary/fixed_vocab_top5k_t1s2.txt'

    with open(grounding_anno_path,'r') as file:
          grounding_anno = json.load(file)

    with open(qa_path,'r') as file:
        qa_file = json.load(file)

    # obtain vocabulary
    with open(vocab_path, 'r') as f:
        vocab_file = f.readlines()
    vocab_file = [line.rstrip('\n') for line in vocab_file]
    vocab_list = list(vocab_file)


    total_qa_acc, total_iou_acc,total_gqa_acc,total_upper_ans, total_upper_anls = [], [], [], [], []
    for i,qa in enumerate(qa_file['data']):
        qa_ans, gd_ans, gqa_ans = 0, 0 ,0

        v_id = qa['video_id']
        q_id = qa['question_id']
        video_path = os.path.join(ocr_file_path, v_id+'.npy')
        ocr_infos = np.load(video_path, allow_pickle=True).item()

        # obtain GT answer 
        gt_ans = qa['answers']

        # obtain GT box
        for i, gt_element in enumerate(grounding_anno['data']):
            if gt_element['question_id'] == q_id:
                gt_info = gt_element['spatial_temporal_gt']
                for j, span in enumerate(gt_info):
                    gt_frame = span['temporal_gt']
                    gt_box = span['bbox_gt']
                    # st = int(gt_frame[0]*10)+1
                    # ed = int(gt_frame[1]*10)+1

        # obtain ocr token list
        ocr_list = []
        frame_list = list(ocr_infos.keys())
        for _,frame in enumerate(ocr_infos):
            ocr_text = ocr_infos[frame]
            for idx, ocr in enumerate(ocr_text):   
                ocr_list.append(ocr['ocr'])

        # QA upper bound: 
        # QA: gt words come from vocab & ocr token; 
        ans_list = vocab_list
        ans_list.extend(ocr_list)
        for ans in gt_ans:
            word = ans.split()
            word_flag = True
            for w in word:
                if w.lower() not in ans_list: 
                    word_flag = False
            if word_flag:
                qa_ans = 1
            else:
                qa_ans = 0


        # ANLS upper bound: 
        anls_score = 0
        anls_score = [max([get_anls(ans, gt) for gt in ans_list]) for ans in gt_ans]  
        

        # # IoU upper bound: 
        # # IoU: all detected box
        detected_box = {key: [] for key in frame_list}
        for fid in frame_list:
            for box in ocr_infos[fid]: 
                x1, y1, x2, y2, x3, y3, x4, y4 = box["points"]
                min_x = min(x1, x2, x3, x4)
                max_x = max(x1, x2, x3, x4)
                min_y = min(y1, y2, y3, y4)
                max_y = max(y1, y2, y3, y4)
                detected_box[fid].append([min_x, min_y, max_x, max_y])

        # IoU
        pred_frame = list(detected_box.keys())
        gt_frame = list(gt_box.keys())
        iou_flag = False
        for fid in pred_frame:
            new_fid = str(int(fid)-1)
            if new_fid in gt_frame:
                pred_b = detected_box[fid]
                gt_b = gt_box[new_fid]
                for box in pred_b:
                    assert box[0]<=box[2] and box[1]<=box[3]
                    result = check_iou(box, gt_b, threshold=0.5) 
                    if result:
                        iou_flag = True
                        continue

        if not iou_flag:
            iou_ans = 0
        else:
            iou_ans = 1

        # GQA
        if qa_ans and iou_ans:
            gqa_ans = 1
        else:
            gqa_ans = 0
        
        total_upper_anls.extend(anls_score)
        total_upper_ans.append(qa_ans)
        total_iou_acc.append(iou_ans)
        total_gqa_acc.append(gqa_ans)
    
    import statistics
    qa_acc = statistics.mean(total_upper_ans)
    anls_acc = statistics.mean(total_upper_anls)
    print(f"{set_} set: qa_acc={qa_acc}")
    print(f"{set_} set: anls_acc={anls_acc}")
    iou_acc = statistics.mean(total_iou_acc)
    gqa_acc = statistics.mean(total_gqa_acc)
    print(f"{set_} set: iou_acc={iou_acc}, gqa_acc={gqa_acc}")