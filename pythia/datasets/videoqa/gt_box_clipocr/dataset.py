# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import os
import json
import glob
from PIL import Image
import torchvision.transforms as transforms
from functools import cmp_to_key

from pythia.datasets.vqa.textvqa.dataset import TextVQADataset
from pythia.utils.text_utils import word_tokenize
from pythia.common.sample import Sample
from pythia.utils.objects_to_byte_tensor import enc_obj2bytes
from transformers import T5Tokenizer, BertTokenizer
from pythia.utils.general import get_pythia_root
from transformers import ViTImageProcessor, ViTModel
import random

import torch.nn.functional as F



class GTBOX(TextVQADataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(
            dataset_type, imdb_file_index, config, *args, **kwargs
        )
        self._name = "gt_box"
        imdb_files = self.config.imdb_files
        if dataset_type not in imdb_files:
            raise ValueError(
                "Dataset type {} is not present in "
                "imdb_files of dataset config".format(dataset_type)
            )

        self.imdb_file = imdb_files[dataset_type][imdb_file_index]
        self.imdb_file = self._get_absolute_path(self.imdb_file)
        self.imdb = np.load(self.imdb_file, allow_pickle=True)[1:]

        file_name = self.imdb_file.split('/')[-1]
        if 'train' in file_name:
            self.split = 'train'
        else:
            self.split = 'test'

        ocr_infos = self.config.ocr_infos
        self.ocr_info_dir = self._get_absolute_path(ocr_infos[dataset_type])
        # self.ocr_frcn_dir = self._get_absolute_path("ocr_en_frcn_features_x101")

        self.num_frames = self.config.frames
        self.frame_ocr_num = self.config.ocr_frame_num
        self.max_ocr_num = self.config.ocr_max_num


        self.processor = ViTImageProcessor.from_pretrained('../huggingface/vit-large-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('../huggingface/vit-large-patch16-224-in21k')
        self.bert_tokenizer = BertTokenizer.from_pretrained('../huggingface/bert-base-uncased')


        #-------------load grounding info---------------------------
        if dataset_type == 'val':
            self.ground_info_dir = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/annotation_bbox_ClipOCR_recog_t1s2val.npy'
        else:
            self.ground_info_dir = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/annotation_bbox_ClipOCR_recog_t1s2test.npy'
        self.ground_info = np.load(self.ground_info_dir, allow_pickle=True)[1:]
        #-------------end---------------------------

    def _get_absolute_path(self, paths):
        if isinstance(paths, list):
            return [self._get_absolute_path(path) for path in paths]
        elif isinstance(paths, str):
            if not os.path.isabs(paths):
                pythia_root = get_pythia_root()
                paths = os.path.join(pythia_root, self.config.data_root_dir, paths)
            return paths
        else:
            raise TypeError(
                "Paths passed to dataset should either be " "string or list"
            )
    
    def __len__(self):
        return len(self.imdb)

    def get_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )
        if isinstance(sample_info["video_id"], int):
            current_sample.image_id = str(sample_info["video_id"])
        else:
            current_sample.image_id = sample_info["video_id"]
        current_sample = self.add_sample_details(sample_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)

        return current_sample

    def add_sample_details(self, sample_info, sample):
        # question
        question_str = sample_info['question']
        processed_question = self.text_processor({"question": question_str})
        sample.text = processed_question['token_inds']
        sample.text_len = processed_question['token_num']

        # process ocr first
        video = sample_info["video_id"]
        width = sample_info['video_width']
        height = sample_info['video_height']

        # load ocr info
        ocr_info_path = os.path.join(self.ocr_info_dir[0], video + ".npy")
        ocr_info = np.load(ocr_info_path, allow_pickle=True).item()

        # sample video frame
        video_path = os.path.join('/data/zsheng/Data_T5_ViteVQA/data/fps10_frames',video)
        frame_paths = glob.glob(os.path.join(video_path,"*.jpg"))

        # obtain frames when setting video fps=10 
        # frame id star with 1
        idxs_list = []
        for i in range(1, len(frame_paths)+1):
            idxs_list.append(i)

        idxs = sample_frames(idxs_list, self.num_frames ,sample='uniform')  # ids对应视频帧序号   

        '''obtain annotated frame & ocr'''
        # start-----------------------------------------------grounding gt for qa
        ground_fid_list = []       
        flag = False     
        for element in self.ground_info:
            if element['question_id'] == sample_info['question_id']:
                for span1 in element['spatial_temporal_gt']:
                    temporal_gt = span1['temporal_gt']
                    fps = int(element['fps'])
                    star_id = int(temporal_gt[0]*fps)+1
                    end_id = int(temporal_gt[1]*fps)+1
                    ground_fid_list.extend(list(range(star_id, end_id + 1)))
                    flag = True
        if not flag:
            ground_fid_list.append(0)
        ground_fid_list = ground_fid_list[:self.num_frames]

        anno_ocr_bbox = np.zeros((0, 4), dtype=np.float32)
        anno_ocr_list, anno_ocr_bbox_list, anno_ocr_track_list, anno_ocr_temporal_list, anno_ocr_mask_list = [], [], [], [], []
        # obtain annotated ocr info
        q_flag = False
        for element in self.ground_info:
            if element['question_id'] == sample_info['question_id']:
                q_flag = True
                # merge multiple temporal span
                human_frame_list, human_ocr_list, human_box_dict, human_ocr_dict = [], [], {}, {}
                for span2 in element['spatial_temporal_gt']:
                    st = int(span2['temporal_gt'][0]*10)
                    ed = int(span2['temporal_gt'][1]*10)
                    if len(set(human_frame_list) & set(list(range(st, ed+1)))) > 0:
                        continue
                    human_frame_list.extend(list(span2['bbox_gt'].keys()))
                    human_box_dict.update(span2['bbox_gt'])
                    human_ocr_list.extend(span2['ocr'])
                    for i, fid in enumerate(human_frame_list):
                        human_ocr_dict[fid] = human_ocr_list[i]

                ocr_idx = 0
                for _, frame_idx in enumerate(idxs): 
                    flag = False 
                    frame_idx = frame_idx -1
                    if str(frame_idx) in human_frame_list: # there is at most one ocr token in a frame, padding 14
                        flag = True
                        ocr_list = human_ocr_dict[str(frame_idx)]
                        points = human_box_dict[str(frame_idx)]
                        x1, y1, x2, y2 = points[0], points[1], points[2], points[3]           
                        anno_ocr_list.append(ocr_list)
                        ocr_idx += 1
                        anno_ocr_bbox_list.append([x1, y1, x2, y2])
                        anno_ocr_track_list.append(frame_idx+1)
                        anno_ocr_temporal_list.append(frame_idx+1)   # ocr temporal id in [1, frame_num], corresponding with grounded frame id. 
                        anno_ocr_mask_list.append(1)

                        padding_size = self.frame_ocr_num - 1
                        if padding_size > 0:
                            # final ocr list
                            anno_ocr_list.extend(["<pad>" for _ in range(padding_size)])
                            anno_ocr_bbox_list.extend([[0, 0, 0, 0] for _ in range(padding_size)])
                            anno_ocr_track_list.extend([frame_idx+1 for _ in range(padding_size)])
                            anno_ocr_temporal_list.extend([frame_idx+1 for _ in range(padding_size)])
                            anno_ocr_mask_list.extend([0 for _ in range(padding_size)])

                    if not flag:   # padding
                        padding_size = self.frame_ocr_num
                        if padding_size > 0:
                            anno_ocr_list.extend(["<pad>" for _ in range(padding_size)])
                            anno_ocr_bbox_list.extend([[0, 0, 0, 0] for _ in range(padding_size)])
                            anno_ocr_track_list.extend([0 for _ in range(padding_size)])
                            anno_ocr_temporal_list.extend([0 for _ in range(padding_size)])
                            anno_ocr_mask_list.extend([0 for _ in range(padding_size)])
                if len(anno_ocr_track_list) == 960:
                    break
            if q_flag:
                break

        # process track_id / temporal_id
        anno_track_embedding = torch.zeros(self.max_ocr_num)
        for i,t_id in enumerate(anno_ocr_track_list):
            anno_track_embedding[i] = t_id
            
        # ocr temporal id
        anno_temporal_embedding = torch.zeros(self.max_ocr_num)
        for i,t_id in enumerate(anno_ocr_temporal_list):
            anno_temporal_embedding[i] = t_id
        
        # ocr temporal id
        anno_temporal_id_embedding = torch.zeros(self.num_frames)
        for i,t_id in enumerate(ground_fid_list):
            anno_temporal_id_embedding[i] = t_id

        
        # input all the annotated frames into the answer decoder
        anno_frame_mask_embedding = torch.zeros(self.num_frames)
        for i in range(self.num_frames):
            if i+1 in ground_fid_list:
                anno_frame_mask_embedding[i] = 1

        # input all the annotated boxes into the answer decoder
        anno_ocr_mask_embedding = torch.zeros(self.max_ocr_num)
        for i,o_id in enumerate(anno_ocr_mask_list):
            anno_ocr_mask_embedding[i] = o_id
        
        if len(anno_ocr_bbox_list) != 0:
            anno_ocr_bbox = np.asarray(anno_ocr_bbox_list[:self.max_ocr_num], dtype=np.float32)
        anno_ocr_bbox_list = anno_ocr_bbox * [1./width, 1./height, 1./width, 1./height]
        # for ocr grounding evaluation
        sample.ocr_bbox_list = self.copy_processor(
            {"blob": anno_ocr_bbox_list.astype(np.float32)}
        )["blob"]
        sample.frame_list = anno_temporal_id_embedding  # for frame grounding evaluation
        sample.frame_mask_embedding = anno_frame_mask_embedding.long()  # for answer decoder
        sample.ocr_mask_embedding = anno_ocr_mask_embedding.long() # for answer decoder
        sample.ocr_track_id = anno_track_embedding.long()
        sample.ocr_temporal_id = anno_temporal_embedding.long()

        # Preprocess OCR tokens
        anno_ocr_list = anno_ocr_list[:self.max_ocr_num]
        # ocr padding
        anno_ocr_tokens = [
            self.ocr_token_processor({"text": token})["text"]
            for token in anno_ocr_list
        ]
        sample_info["ocr_tokens"] = anno_ocr_tokens

        # Get FastText embeddings for OCR tokens
        context = self.context_processor({"tokens": anno_ocr_tokens})
        sample.context = context["text"]
        sample.context_tokens = context["tokens"]
        sample.context_tokens_enc = enc_obj2bytes(sample.context_tokens) 
        sample.context_feature_0 = context["text"]
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = context["length"]

        # Get PHOC embeddings for OCR tokens
        context_phoc = self.phoc_processor({"tokens": anno_ocr_tokens})
        sample.context_feature_1 = context_phoc["text"]
        sample.context_info_1 = Sample()
        sample.context_info_1.max_features = context_phoc["length"]

        # end-----------------------------------------------grounding gt for qa


        '''obtain detected frame & ocr'''
        # start-------------------------------------------------------
        ocr_bbox = np.zeros((0, 4), dtype=np.float32)
        frame_id_list, frame_mask_list = [], []
        ocr_list, ocr_bbox_list, ocr_track_list, ocr_temporal_list, ocr_mask_list = [], [], [], [], []
        for idx, frame_idx in enumerate(idxs):
             
            frame_ocr, frame_bbox, frame_track, frame_temporal, ocr_frame_mask = [], [], [], [], []
            if len(ocr_info) >= frame_idx:
                frame_result = ocr_info[str(frame_idx)]
            else:
                frame_result = ocr_info[str(frame_idx-1)]
            for _, ocr_result_frame in enumerate(frame_result):
                points = ocr_result_frame['points']
                x1 = min(points[0],points[6])
                y1 = min(points[1],points[3])
                x2 = max(points[2],points[4])
                y2 = max(points[5],points[7])

                frame_ocr.append(ocr_result_frame['ocr'])
                frame_bbox.append([x1, y1, x2, y2])
                frame_track.append(ocr_result_frame['ID'])
                frame_temporal.append(frame_idx)   # ocr temporal id in [1, frame_num], corresponding with grounded frame id. 
                ocr_frame_mask.append(1)
                
            padding_size = self.frame_ocr_num - len(frame_ocr)
            if padding_size > 0:
                frame_ocr.extend(["<pad>" for _ in range(padding_size)])
                frame_bbox.extend([[0, 0, 0, 0] for _ in range(padding_size)])
                frame_track.extend([0 for _ in range(padding_size)])
                frame_temporal.extend([frame_idx for _ in range(padding_size)])
                ocr_frame_mask.extend([0 for _ in range(padding_size)])
            else:
                frame_ocr = frame_ocr[:self.frame_ocr_num]
                frame_bbox = frame_bbox[:self.frame_ocr_num]
                frame_track = frame_track[:self.frame_ocr_num]
                frame_temporal = frame_temporal[:self.frame_ocr_num]
                ocr_frame_mask = ocr_frame_mask[:self.frame_ocr_num]

            ocr_list.extend(frame_ocr)
            ocr_bbox_list.extend(frame_bbox)
            ocr_track_list.extend(frame_track)
            ocr_temporal_list.extend(frame_temporal)
            ocr_mask_list.extend(ocr_frame_mask)
            frame_id_list.append(frame_idx)
            frame_mask_list.append(1)

        ocr_track_list = ocr_track_list[:self.max_ocr_num]
        ocr_temporal_list = ocr_temporal_list[:self.max_ocr_num]
        ocr_mask_list = ocr_mask_list[:self.max_ocr_num]

        frame_padding_size = self.num_frames - len(idxs)
        if frame_padding_size > 0:
            frame_id_list.extend([0 for _ in range(frame_padding_size)])
            frame_mask_list.extend([0 for _ in range(frame_padding_size)])
        frame_id_list = frame_id_list[:self.num_frames]
        frame_mask_list = frame_mask_list[:self.num_frames]


        if len(ocr_bbox_list) != 0:
            ocr_bbox = np.asarray(ocr_bbox_list[:self.max_ocr_num], dtype=np.float32)

        # normalized bbox
        ocr_bbox_list = ocr_bbox * [1./width, 1./height, 1./width, 1./height]
        # ocr bbox padding
        sample.ocr_bbox_coordinates = self.copy_processor(
            {"blob": ocr_bbox_list.astype(np.float32)}
        )["blob"]
        
        # process track_id / temporal_id
        track_embedding = torch.zeros(self.max_ocr_num)
        for i,t_id in enumerate(ocr_track_list):
            track_embedding[i] = t_id

        temporal_embedding = torch.zeros(self.max_ocr_num)
        for i,t_id in enumerate(ocr_temporal_list):
            temporal_embedding[i] = t_id
        
        frame_embedding = torch.zeros(self.num_frames)
        for i,f_id in enumerate(frame_id_list):
            frame_embedding[i] = f_id

        ocr_mask_embedding = torch.zeros(self.max_ocr_num)
        for i,m_id in enumerate(ocr_mask_list):
            ocr_mask_embedding[i] = m_id

        frame_mask_embedding = torch.zeros(self.num_frames)
        for i,f_id in enumerate(frame_mask_list):
            frame_mask_embedding[i] = f_id
        

        # sample.track_id = track_embedding.long()
        # sample.temporal_id = temporal_embedding.long()
        # sample.ocr_mask = ocr_mask_embedding.long()
        sample.frame_id = frame_embedding.long()
        sample.frame_mask = frame_mask_embedding.long()
        
        
        # Preprocess OCR tokens
        # ocr_list = ocr_list[:self.max_ocr_num]
        # # ocr padding
        # ocr_tokens = [
        #     self.ocr_token_processor({"text": token})["text"]
        #     for token in ocr_list
        # ]
        # sample_info["ocr_tokens"] = ocr_tokens

        # # Get FastText embeddings for OCR tokens
        # context = self.context_processor({"tokens": ocr_tokens})
        # sample.context = context["text"]
        # sample.context_tokens = context["tokens"]
        # sample.context_tokens_enc = enc_obj2bytes(sample.context_tokens) 
        # sample.context_feature_0 = context["text"]
        # sample.context_info_0 = Sample()
        # sample.context_info_0.max_features = context["length"]

        # # Get PHOC embeddings for OCR tokens
        # context_phoc = self.phoc_processor({"tokens": ocr_tokens})
        # sample.context_feature_1 = context_phoc["text"]
        # sample.context_info_1 = Sample()
        # sample.context_info_1.max_features = context_phoc["length"]

        # # end---------------------------------------------------------------------------
        

        # use ViT-L to extract frame features
        imgs = []
        vit_feat_path = os.path.join('/data/zsheng/Data_T5_ViteVQA/data/fps10_video_vit_feat',video)
        for idx in idxs:
            v_path = os.path.join(vit_feat_path, str(idx)+'.npy')
            video_feat = np.load(v_path, allow_pickle=True)
            imgs.append(video_feat)  # [197, 1024] -> [1, 1024]
   
        imgs = torch.from_numpy(np.concatenate(imgs, axis=0))  # [frame_nums, 1024]
        sample.video_feat = F.pad(imgs, (0, 0, 0, self.num_frames - imgs.size(0)))

        sample.frame_num = torch.tensor(len(idxs) , dtype=torch.int64)
        sample.global_q = torch.tensor(1, dtype=torch.int64)

        return sample


    def add_answer_info(self, sample_info, sample):
        answers_list = sample_info["answers"]
        random.shuffle(answers_list)
        answers_list = answers_list[:2]
        if len(answers_list) == 1:
            answers = answers_list * 10
        else:
            answers = [answers_list[0]] * 5 + [answers_list[1]] * 5
        sample.gt_answers_enc = enc_obj2bytes(answers)
        answer_processor_arg = {
            "answers": answers,
            "context_tokens": sample.context_tokens,
        }
        processed_answers = self.answer_processor(answer_processor_arg)
        assert not self.config.fast_read, \
            'In M4CTextVQADataset, online OCR sampling is incompatible' \
            'with fast_read, so fast_read is currently not supported.'
        sample.targets = processed_answers["answers_scores"]
        sample.sampled_idx_seq = processed_answers["sampled_idx_seq"]
        sample.train_prev_inds = processed_answers["train_prev_inds"]
        sample.train_loss_mask = processed_answers["train_loss_mask"]

        return sample


    def format_for_evalai(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()
        ground_frames = report.ground_frame.tolist()
        ground_boxs = report.ground_box.tolist()

        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            context_tokens = report.context_tokens[idx]
            answer_words = []
            pred_source = []
            ground_frame = ground_frames[idx]
            ground_box = ground_boxs[idx]

            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(
                        word_tokenize(context_tokens[answer_id])
                    )
                    pred_source.append('OCR')
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )
                    pred_source.append('VOCAB')
                
            # join all the answer tokens with space
            # (this should be correct for almost all cases)
            pred_answer = ' '.join(answer_words).replace(" 's", "'s")
            entry = {
                "question_id": question_id.item(),
                "video_id": report.image_id[idx],
                "answer": pred_answer,
                "grounded frame": ground_frame,
                "grounded box": ground_box,
                "pred_source": pred_source,
            }
            entry = self.postprocess_evalai_entry(entry)

            predictions.append(entry)

        return predictions
    def preprocess_sample_info(self, sample_info):
        return sample_info  # Do nothing

    def postprocess_evalai_entry(self, entry):
        return entry  # Do nothing


def sample_frames(frames, sample_len, sample='uniform'):

    step = len(frames) // sample_len

    if len(frames) <= sample_len:
        sample_id_list = frames
    else:
        step = len(frames) // sample_len  # 步长
        sample_id_list = [frames[i * step] for i in range(sample_len)]

    return sample_id_list