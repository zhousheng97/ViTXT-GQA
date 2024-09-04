# Copyright (c) Facebook, Inc. and its affiliates.
import re
import torch

class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/pythia/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile("(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile("(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


class TextVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers):
        """
        compute the accuracy (soft score) of human answers
        """
        answers = [self.answer_processor(a) for a in raw_answers]
        assert len(answers) == 10
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        unique_answer_scores = {}

        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [
                    item for item in gt_answers if item != gt_answer
                ]
                matching_answers = [
                    item for item in other_answers if item[1] == unique_answer
                ]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores

    def eval_pred_list(self, pred_scores, pred_list):
        
        for entry in pred_list:
            pred_answer = self.answer_processor(entry['pred_answer'])
            unique_answer_scores = self._compute_answer_scores(
                entry['gt_answers']
            )
            score = unique_answer_scores.get(pred_answer, 0.)
            pred_scores.append(score)

        accuracy = sum(pred_scores) / len(pred_scores)
        return pred_scores, accuracy


class STVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def eval_pred_list(self, pred_scores, pred_list):
        for entry in pred_list:
            pred_answer = self.answer_processor(entry['pred_answer'])
            gts = [self.answer_processor(a) for a in entry['gt_answers']]
            score = (1. if pred_answer in gts else 0.)
            pred_scores.append(score)

        accuracy = sum(pred_scores) / len(pred_scores)
        return pred_scores, accuracy


class STVQAANLSEvaluator:
    def __init__(self):
        import editdistance  # install with `pip install editdistance`
        self.get_edit_distance = editdistance.eval

    def get_anls(self, s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        iou = 1 - self.get_edit_distance(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= .5 else 0.
        return anls

    def eval_pred_list(self, pred_scores, pred_list):
        for entry in pred_list:
            anls = max(
                self.get_anls(entry['pred_answer'], gt)
                for gt in entry['gt_answers']
            )
            pred_scores.append(anls)

        accuracy = sum(pred_scores) / len(pred_scores)
        return pred_scores, accuracy


class TempGroundAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()
    
    def eval_pred_list(self, pred_list):
        pred_scores = []
        for entry in pred_list:
            flag = False
            pred_frame = entry['pred_frame']
            gt_frame = entry['st_gt']
            video_fps = entry['video_fps']
            for _, t_span in enumerate(gt_frame):
                temporal_gt = t_span['temporal_gt']
                st_frame = int(temporal_gt[0] * video_fps)+1
                ed_frame = int(temporal_gt[1] * video_fps)+1

                if any(st_frame <= element <= ed_frame for element in pred_frame):
                    pred_scores.append(1)
                    flag = True
                    break
            if not flag: 
                pred_scores.append(0)

                
        t_accuracy = sum(pred_scores) / len(pred_scores)
        return t_accuracy
    

class BoxGroundAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def calculate_iou(self, box1, box2):
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
    
    def check_iou(self, pred_scores, bbox_iou_value, gt_bbox, pred_bboxs, threshold = None):
        flag = False
        bbox_iou = -1
        max_iou = 0
        for _,pred_bbox in enumerate(pred_bboxs):
            assert pred_bbox[0]<=pred_bbox[2] and pred_bbox[1]<=pred_bbox[3]
            bbox_iou = self.calculate_iou(gt_bbox, pred_bbox)
            if bbox_iou > max_iou:
                max_iou = bbox_iou

        if max_iou > threshold:
            flag = True
            
        bbox_iou_value.append(bbox_iou)
        if flag:        
            pred_scores.append(1)           
            
        return pred_scores, bbox_iou_value, flag
    
    def eval_pred_list(self, pred_scores, pred_list, threshold=None):
        pred_box = []
        for entry in pred_list:
            width, height = entry['width'], entry['height']
            pred_boxs = [[bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height] for bbox in entry['pred_box']]

            pred_frames = entry['pred_frame']
            gt_frame = entry['st_gt']
            video_fps = entry['video_fps']
            ocr_topk = entry['ocr_topk']
            bbox_iou_value = []
            flag = False
            for _, t_span in enumerate(gt_frame):
                temporal_gt = t_span['temporal_gt']
                bboxs_gt = t_span['bbox_gt']
                st_frame = int(temporal_gt[0] * video_fps)+1
                ed_frame = int(temporal_gt[1] * video_fps)+1
                for id, frame_idx in enumerate(pred_frames):
                    if st_frame <= int(frame_idx) <= ed_frame:  # calculate IoU between pred_bbox and gt_box in pred_frame
                        if str(int(frame_idx-1)) in bboxs_gt.keys():
                            bbox_gt = bboxs_gt[str(int(frame_idx-1))]
                            # obtain the topk ocr in the grounded frame
                            pred_box = pred_boxs[id*ocr_topk:(id+1)*ocr_topk]
                            assert bbox_gt[0]<=bbox_gt[2] and bbox_gt[1]<=bbox_gt[3]
                            pred_scores, bbox_iou_value, flag = self.check_iou(pred_scores, bbox_iou_value, bbox_gt, pred_box, threshold) 
                    
            if not flag: 
                pred_scores.append(0)

        box_accuracy = sum(pred_scores) / len(pred_scores)
        return pred_scores, box_accuracy