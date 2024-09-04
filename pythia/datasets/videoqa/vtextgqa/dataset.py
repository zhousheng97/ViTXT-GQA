# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import os
import glob

from pythia.datasets.vqa.textvqa.dataset import TextVQADataset
from pythia.utils.text_utils import word_tokenize
from pythia.common.sample import Sample
from pythia.utils.objects_to_byte_tensor import enc_obj2bytes
from transformers import BertTokenizer
from pythia.utils.general import get_pythia_root
from transformers import ViTImageProcessor, ViTModel
import random
import torch.nn.functional as F



class VTEXTGQADataset(TextVQADataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(
            dataset_type, imdb_file_index, config, *args, **kwargs
        )
        self._name = "vtextgqa"
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

        self.num_frames = self.config.frames
        self.frame_ocr_num = self.config.ocr_frame_num

        self.processor = ViTImageProcessor.from_pretrained('../../huggingface/vit-large-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('../../huggingface/vit-large-patch16-224-in21k')
        self.bert_tokenizer = BertTokenizer.from_pretrained('../../huggingface/bert-base-uncased')

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
        current_sample['gt_answers'] = sample_info['answers']
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

        # obtain frames when video fps=10
        idxs_list = []
        # frame id start from 1
        for i in range(1, len(frame_paths)+1):
            idxs_list.append(i)

        idxs = sample_frames(idxs_list, self.num_frames ,sample='uniform')  # frame ids          

        # Get ocr info for sampled frame
        ocr_bbox = np.zeros((0, 4), dtype=np.float32)

        frame_id_list, frame_mask_list = [], []
        ocr_list, ocr_bbox_list, ocr_track_list, ocr_temporal_list, ocr_mask_list = [], [], [], [], []
        for idx, frame_idx in enumerate(idxs):
             
            frame_ocr, frame_bbox, frame_track, frame_temporal, ocr_frame_mask = [], [], [], [], []
            # ocr detection <-> frame_idx
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
                frame_temporal.append(frame_idx)   
                ocr_frame_mask.append(1)
                
            # ocr padding
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

        ocr_track_list = ocr_track_list
        ocr_temporal_list = ocr_temporal_list
        ocr_mask_list = ocr_mask_list 
        frame_id_list = frame_id_list[:self.num_frames]
        frame_mask_list = frame_mask_list[:self.num_frames]

        # -------- middle_frame --------:
        if max(frame_id_list) <= self.num_frames: 
            # print(frame_id_list)
            new_frame_id_list = frame_id_list
        else:
            new_frame_id_list = frame_id_list[:self.num_frames]
        
        # obtain middle frame
        middel_frame_id_list = []
        # mid
        frame_index = new_frame_id_list[len(new_frame_id_list) // 2] 
        # left
        frame_index = new_frame_id_list[0] 
        # right
        frame_index = new_frame_id_list[-1] 
        # assert 1 <= frame_index <= len(frame_paths)
        middel_frame_id_list.append(frame_index)
        middel_frame_id_embedding = torch.zeros(1)
        middel_frame_idx_embedding = torch.zeros(1)
        for i,mid_id in enumerate(middel_frame_id_list):
            middel_frame_id_embedding[i] = mid_id # start from 1
            if mid_id >= self.num_frames:
                middel_frame_idx_embedding[i] = len(new_frame_id_list)//2 + 1
            else:
                middel_frame_idx_embedding[i] = mid_id
            assert mid_id in new_frame_id_list

        sample.middel_frame_id = middel_frame_id_embedding.long()
        sample.middel_frame_idx = middel_frame_idx_embedding.long()
        # -------------end

        
        # frame padding
        frame_padding_size = self.num_frames - len(idxs)
        if frame_padding_size > 0:
            frame_id_list.extend([0 for _ in range(frame_padding_size)])
            frame_mask_list.extend([0 for _ in range(frame_padding_size)])

        if len(ocr_bbox_list) != 0:
            ocr_bbox = np.asarray(ocr_bbox_list, dtype=np.float32)

        # normalized bbox
        ocr_bbox_list = ocr_bbox * [1./width, 1./height, 1./width, 1./height]
        # ocr bbox padding
        sample.ocr_bbox_coordinates = self.copy_processor(
            {"blob": ocr_bbox_list.astype(np.float32)}
        )["blob"]

      
        # process track_id / temporal_id
        track_embedding = torch.zeros(self.frame_ocr_num*self.num_frames)
        for i,t_id in enumerate(ocr_track_list):
            track_embedding[i] = t_id

        temporal_embedding = torch.zeros(self.frame_ocr_num*self.num_frames)
        for i,t_id in enumerate(ocr_temporal_list):
            temporal_embedding[i] = t_id
        
        frame_embedding = torch.zeros(self.num_frames)
        for i,f_id in enumerate(frame_id_list):
            frame_embedding[i] = f_id

        ocr_mask_embedding = torch.zeros(self.frame_ocr_num*self.num_frames)
        for i,m_id in enumerate(ocr_mask_list):
            ocr_mask_embedding[i] = m_id

        frame_mask_embedding = torch.zeros(self.num_frames)
        for i,f_id in enumerate(frame_mask_list):
            frame_mask_embedding[i] = f_id
        

        sample.track_id = track_embedding.long()
        sample.temporal_id = temporal_embedding.long()
        sample.ocr_mask = ocr_mask_embedding.long()
        sample.frame_id = frame_embedding.long()
        sample.frame_mask = frame_mask_embedding.long()
        
        # Preprocess OCR tokens
        ocr_list = ocr_list[:self.frame_ocr_num*self.num_frames]
        # ocr padding
        ocr_tokens = [
            self.ocr_token_processor({"text": token})["text"]
            for token in ocr_list
        ]
        sample_info["ocr_tokens"] = ocr_tokens

        # Get FastText embeddings for OCR tokens
        context = self.context_processor({"tokens": ocr_tokens})
        sample.context = context["text"]
        sample.context_tokens = context["tokens"]
        sample.context_tokens_enc = enc_obj2bytes(sample.context_tokens) 
        sample.context_feature_0 = context["text"]
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = context["length"]

        # Get PHOC embeddings for OCR tokens
        context_phoc = self.phoc_processor({"tokens": ocr_tokens})
        sample.context_feature_1 = context_phoc["text"]
        sample.context_info_1 = Sample()
        sample.context_info_1.max_features = context_phoc["length"]
        
        # use ViT-L to extract frame features
        imgs = []
        vit_feat_path = os.path.join('/data/zsheng/Data_T5_ViteVQA/data/fps10_video_vit_feat',video)
        for idx in idxs:
            v_path = os.path.join(vit_feat_path, str(idx)+'.npy')
            video_feat = np.load(v_path, allow_pickle=True)
            imgs.append(video_feat)  # [197, 1024] -> [1, 1024]
            if idx == middel_frame_id_list[0]:
                mid_img_feat = np.load(v_path, allow_pickle=True)
        
        # obtain middle frame feat
        sample.mid_img_feat = torch.from_numpy(mid_img_feat)  # [1, 1024]

        # video feat
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
        step = len(frames) // sample_len 
        sample_id_list = [frames[i * step] for i in range(sample_len)]

    return sample_id_list