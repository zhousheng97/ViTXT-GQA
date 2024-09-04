# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import math
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from einops import rearrange

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)
from einops import rearrange, repeat
from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.layers import ClassifierLayer
from transformers import DistilBertConfig

import torch.nn as nn
# from pythia.modules.mist_module.language_model import Bert, AModel
# from transformers import DistilBertConfig, BertConfig
import torch.nn.functional as F
# from pythia.modules.mist_module.EncoderVid import EncoderVid
from pythia.modules.mist_module.mist_module import Embeddings, PositionEmbeddings, TokenTypeEmbeddings, ISTA, Transformer
from pythia.modules.mist_module.clip import clip


@registry.register_model("mist")
class MIST(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer_config = BertConfig(**self.config.translayers) 
        self.grounding_config = self.config.grounding  
        self.bert_config = BertConfig(**self.config.encoder)
        self.mmt_config = BertConfig(**self.config.mmt)
        self._datasets = registry.get("config").datasets.split(",")

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_model()
        self._build_mmt()
        self._build_output()

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.config.text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                '../../huggingface/bert-base-uncased', config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append({
                'module': self.text_bert,
                'lr_scale': self.config.lr_scale_text_bert,
            })
        else:
            self.writer.write('NOT initializing text_bert from BERT_BASE')
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            self.writer.write(
                'Projecting text_bert output to {} dim'.format(
                    self.mmt_config.hidden_size
                )
            )
            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):

        self.frame_embeddings = nn.Embedding(
            4000, 50
        )
        
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.obj_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_frame_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )
        self.linear_obj_frame_to_mmt_in = nn.Linear(
            50, self.mmt_config.hidden_size
        )
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)

    def _build_ocr_encoding(self):

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )
        self.spatial_enhance = nn.LSTM(num_layers=2,input_size=300,hidden_size=300,batch_first=True,bidirectional=True) # (batch_size, sequence_length, 600)
        # OCR location feature: relative bounding box coordinates (4-dim)
        # 8 dim st, ed
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )
        # for track_id
        # for temporal_id
        # TODO
        self.temporal_position_embeddings = nn.Embedding(
            4000, 50
        )
        self.track_position_embeddings = nn.Embedding(
            4000, 50
        )

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append({
            'module': self.mmt,
            'lr_scale': self.config.lr_scale_mmt,
        })

    def _build_model(self):
        self.VideoQAmodel = MIST_VideoQA(self.grounding_config)

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)

        # fixed answer vocabulary scores
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)

        num_choices -= self.config.classifier.ocr_max_num
        self.classifier = ClassifierLayer(
            self.config["classifier"]["type"],
            in_dim=self.mmt_config.hidden_size,
            out_dim=num_choices,
            **self.config["classifier"]["params"]
        )

        self.answer_processor = registry.get(
            self._datasets[0] + "_answer_processor"
        )

    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(sample_list, fwd_results)
        self._forward_obj_encoding(sample_list, fwd_results)
        self._forward_ocr_encoding(sample_list, fwd_results)
        self._forward_pam_graph(sample_list, fwd_results)
        self._forward_mmt_and_output(sample_list, fwd_results)

        # only keep scores in the forward pass results
        
        results = {
            "pos_scores": fwd_results["pos_scores"], 
            "ground_box": fwd_results['ground_bbox'], 
            "ground_frame": fwd_results['ground_frame'],
            "frame_topk": fwd_results['frame_topk'], 
            "ocr_topk": fwd_results['ocr_topk'],
            }

        return results
    
    def _forward_txt_encoding(self, sample_list, fwd_results):
        fwd_results['txt_inds'] = sample_list.text

        # binary mask of valid text (question words) vs padding
        fwd_results['txt_mask'] = _get_mask(
            sample_list.text_len, sample_list.text.size(1)
        )

        # first forward the text BERT layers
        text_bert_out = self.text_bert(
            txt_inds=fwd_results['txt_inds'],
            txt_mask=fwd_results['txt_mask']
        )
        fwd_results['txt_emb'] = self.text_bert_out_linear(text_bert_out)
    
    def _forward_obj_encoding(self, sample_list, fwd_results):

        # load frame features extracted by ViT
        video_frcn = sample_list.video_feat
        video_frcn = F.normalize(video_frcn, dim=-1)
        assert video_frcn.size(-1) == 1024
        
        # load frame id embedding
        frame_id = sample_list.frame_id
        video_temporal_id = self.frame_embeddings(frame_id)
        assert video_temporal_id.size(-1) == 50

        video_mmt_in = torch.cat(
            [video_frcn, video_temporal_id],
            dim=-1
        )

        video_mmt_in = (
            self.obj_feat_layer_norm(
                self.linear_obj_feat_to_mmt_in(video_mmt_in)
                )
        )
        video_mmt_in = self.obj_drop(video_mmt_in)
        fwd_results['obj_mmt_in'] = video_mmt_in

        # binary mask of valid object vs padding
        frame_mask = sample_list.frame_mask
        fwd_results['obj_mask'] = frame_mask

    def _forward_ocr_encoding(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        
        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_temporal_id = sample_list.temporal_id
        ocr_temporal_id = self.temporal_position_embeddings(ocr_temporal_id)
        ocr_track_id = sample_list.track_id
        ocr_track_id = self.track_position_embeddings(ocr_track_id)
        ocr_bbox = sample_list.ocr_bbox_coordinates

      
        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_temporal_id, ocr_track_id],
            dim=-1
        )
        
        ocr_mmt_in = (
            self.ocr_feat_layer_norm(
                self.linear_ocr_feat_to_mmt_in(ocr_feat)
            ) + self.ocr_bbox_layer_norm(
                self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
            )
        )
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results['ocr_mmt_in'] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_mask = sample_list.ocr_mask
        fwd_results['ocr_mask'] = ocr_mask
        

    def _forward_pam_graph(self, sample_list, fwd_results):
        self.VideoQAmodel(sample_list, fwd_results)

    def _forward_mmt(self, sample_list, fwd_results):
        # first forward the text BERT layers
        text_bert_out = self.text_bert(
            txt_inds=fwd_results['txt_inds'],
            txt_mask=fwd_results['txt_mask']
        )
        fwd_results['txt_emb'] = self.text_bert_out_linear(text_bert_out)

        mmt_results = self.mmt(
            txt_emb=fwd_results['txt_emb'],
            txt_mask=fwd_results['txt_mask'],
            obj_emb=fwd_results['obj_mmt_in'],
            obj_mask=fwd_results['obj_mask'],
            ocr_emb=fwd_results['ocr_mmt_in'],
            ocr_mask=fwd_results['ocr_mask'],
            fixed_ans_emb=self.classifier.module.weight,
            prev_inds=fwd_results['prev_inds'],
        )
        fwd_results.update(mmt_results)

    def _forward_output(self, sample_list, fwd_results):
        mmt_dec_output = fwd_results['mmt_dec_output']
        mmt_ocr_output = fwd_results['mmt_ocr_output']
        ocr_mask = fwd_results['ocr_mask']

        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(
            mmt_dec_output, mmt_ocr_output, ocr_mask
        )
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        fwd_results['pos_scores'] = scores

    def _forward_mmt_and_output(self, sample_list, fwd_results):
        if self.training:
            fwd_results['prev_inds'] = sample_list.train_prev_inds.clone()
            self._forward_mmt(sample_list, fwd_results)
            self._forward_output(sample_list, fwd_results)
        else:
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results['prev_inds'] = torch.zeros_like(
                sample_list.train_prev_inds
            )
            fwd_results['prev_inds'][:, 0] = self.answer_processor.BOS_IDX

            # greedy decoding at test time
            for t in range(dec_step_num):
                self._forward_mmt(sample_list, fwd_results)
                self._forward_output(sample_list, fwd_results)

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = fwd_results["pos_scores"].argmax(dim=-1)
                fwd_results['prev_inds'][:, 1:] = argmax_inds[:, :-1]

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer_attributes.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * m['lr_scale']
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups
    
class MIST_VideoQA(nn.Module):
    def __init__(
        self,
        config,
        feature_dim=768,
        word_dim=768,
        N=2,
        h=8,
        d_model=768,
        d_ff=2048,
        dropout=0.1,
        Q=20,
        T=20,
        vocab_size=30522,
        baseline="",
        n_negs=1,
        probe=False,
        topk=1,
        numc=16,
        topj=5,
    ):
        """
        :param feature_dim: dimension of the input video features
        :param word_dim: dimension of the input question features
        :param N: number of transformer layers
        :param h: number of transformer heads
        :param d_model: dimension for the transformer and final embedding
        :param d_ff: hidden dimension in the transformer
        :param dropout: dropout rate in the transformer
        :param Q: maximum number of tokens in the question
        :param T: maximum number of video features
        :param vocab_size: size of the vocabulary for the masked language modeling head
        :param baseline: set as "qa" not to use the video
        :param n_negs: number of negatives sampled for cross-modal matching
        :param probe: whether or not to freeze all parameters but the heads
        :param topk: number of segments to select
        :param numc: number of segments per video
        :param topj: number of objects to select
        """
        super().__init__()
        # positional and modality encoding
        self.frame_topk = config.frame_topk
        self.ocr_topk = config.ocr_topk
        self.numc = numc
        self.numf = numc
        # self.numf = int(32 / self.numc)
        T = 32 + (16) * self.frame_topk * self.numf
        self.position = Embeddings(d_model, Q, T, dropout, True)
        self.frame_position_embedding = PositionEmbeddings(512, 32, True)
        self.question_position_embedding = PositionEmbeddings(512, Q, True)
        self.token_type_embedding = TokenTypeEmbeddings(512, 3)
        self.config = config

        # d_pos = 128
        # self.encode_vid = EncoderVid(feat_dim=feature_dim,
        #                              bbox_dim=5,
        #                              feat_hidden=d_model,
        #                              pos_hidden=d_pos)

        self.self_attn = nn.Linear(self.config.hidden_size, 1)

        # video and question fusion modules
        self.ISTA = [ISTA(self.config, feature_dim=feature_dim, word_dim=word_dim, Q=Q, N=N,
                          d_model=d_model, dropout=dropout, d_ff=d_ff, h=h, topk=self.frame_topk, topj=self.ocr_topk)]
        for _ in range(1):
            self.ISTA.append(
                ISTA(self.config, feature_dim=d_model, word_dim=d_model, Q=Q, N=N,
                     d_model=d_model, dropout=dropout, d_ff=d_ff, h=h, topk=self.frame_topk, topj=self.ocr_topk)
            )
        self.ISTA = nn.ModuleList(self.ISTA)

        # answer prediction
        self.vqproj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, d_model))

        # parameters
        self.Q = Q
        self.T = T
        self.n_negs = n_negs

        # cross-modal matching head
        self.crossmodal_matching = nn.Linear(d_model, 1)
        self.cm_loss_fct = nn.BCELoss()

        self.config = DistilBertConfig.from_pretrained(
            "/data/zsheng/huggingface/distilbert-base-uncased",
            n_layers=N,
            dim=d_model,
            dropout=dropout,
            hidden_dim=d_ff,
            attention_dropout=dropout,
            n_heads=h,
        )
        self.ttrans = Transformer(self.config)


        # weight initialization
        self.apply(self._init_weights)
        self.answer_embeddings = None

        # pretrained DistilBERT language model
        # self.bert = Bert()
        self.clip, _ = clip.load("ViT-B/32")

        # answer modules
        # self.amodel = AModel(out_dim=d_model, sentence_dim=2048)

        if probe: # freeze all layers but the heads
            for n, p in self.named_parameters():
                if "vqproj" not in n and (
                    ("amodel" not in n) or ("linear_text" not in n)
                ):
                    p.requires_grad_(False)
                else:
                    print(n)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # def _compute_answer_embedding(self, a2v):
    #     self.answer_embeddings = self.get_answer_embedding(a2v)

    # def get_patch_frame_feat(self, video):
    #     video_o = video[0]
    #     video_f = video[1]
    #     return video_o, video_f

    # def get_answer_embedding(self, answer):
    #     answer = self.amodel(answer)
    #     return answer

    # def get_clip_txt_embedding(self, question):
    #     bsize = question.size(0)
    #     question_clip, word_clip = self.clip.encode_text(question)

    #     question_clip = question_clip / question_clip.norm(dim=-1, keepdim=True)   # [bsize, CLIP_dim]
    #     question_clip = question_clip.view(bsize, -1, 1).float()  # [bsize, 1, CLIP_dim]

    #     word_clip = word_clip / word_clip.norm(dim=-1, keepdim=True)   # [bsize, num_word, CLIP_dim]
    #     word_clip = word_clip.view(bsize, -1, 1).float()  # [bsize, num_word, CLIP_dim]
    #     return question_clip, word_clip

    def _calculate_self_attn(self, ques, mask):
        attn = self.self_attn(ques).squeeze(-1)  # N, 20
        attn = F.softmax(attn, dim=-1)
        attn = attn * mask
        attn = attn / (attn.sum(1, keepdim=True)+ 1e-12)
        question_feature = torch.bmm(attn.unsqueeze(1), ques)  # N, 1, 768
        return question_feature

    def forward(
        self,
        sample_list, 
        fwd_results, 
        mode="vqa",
        
    ):
        """
        :param video: video features
        :param question: [bs, Q]
        :param labels: [bs, Q] used for masked language modeling
        :param answer: [batch_size, amax_words, 300] used for contrastive loss training, otherwise precomputed at the vocabulary level
        :param video_mask: [bs, T]
        :param text_mask: [bs, Q]
        :param numc: number of segments per video
        :param topk: number of segments to select
        :param topj: number of objects to select
        """
        ocr_feat = fwd_results['ocr_mmt_in']
        ocr_mask = sample_list.ocr_mask
        frame_feat = fwd_results['obj_mmt_in']
        frame_mask = fwd_results['obj_mask']
        q_feat = fwd_results['txt_emb']
        q_mask = _get_mask(
            sample_list.text_len, sample_list.text.size(1)
        )
        ocr_box = sample_list.ocr_bbox_coordinates
        device = frame_feat.device

        video_o, video_f = ocr_feat, frame_feat # self.get_patch_frame_feat(frame_feat)
        # video_o: [bs, num_clip * num_frame, num_object, 768]
        # video_f: [bs, num_clip * num_frame, 768])
        bsize, ocr_num, fdim = video_o.size()
        bsize, frame_num, fdim = frame_feat.size()
        numc = self.numc  # clip number
        numf = frame_num // numc  # frame number per clip
        numo = ocr_num // (numc * numf)  # ocr number per frame
        
        # embed video and question
        video_o = video_o.view(bsize, numc * numf, numo, fdim)
        # video_o = self.encode_vid(video_o).view(bsize, numc, numf, numo, -1)

        # q_feat, w_feat = self.get_clip_txt_embedding(question)
        global_q_feat = self._calculate_self_attn(q_feat, q_mask)  # [bs, 1, 768]

        video_f_norm = video_f / video_f.norm(dim=-1, keepdim=True)
        seg_feat = video_f_norm.view(bsize, numc * numf, -1)
        # seg_feat = torch.mean(video_clip, dim=-2) # [bs, clip_num, 768]

        # question = self.bert(question)
        if q_feat.shape[1] < self.Q:
            q_feat = torch.cat(
                [
                    q_feat,
                    torch.zeros(
                        q_feat.shape[0],
                        self.Q - q_feat.shape[1],
                        q_feat.shape[2],
                    ).cuda(),
                ],
                1,
            )
            # text_mask = torch.cat(
            #     [
            #         text_mask,
            #         torch.zeros(
            #             text_mask.shape[0], self.Q - text_mask.shape[1]
            #         ).cuda(),
            #     ],
            #     1,
            # )


        # perform ISTA layers
        # out_list = []
        for ista in self.ISTA:
            question_proj, _, ground_frame_idx, ground_frame_mask, _, ground_ocr_mask = ista(self.config, sample_list, global_q_feat, q_mask, q_feat, seg_feat, video_o)
            # out_list.append(attended_vq)

        # obtain ocr box
        ocr_box = sample_list.ocr_bbox_coordinates
        ground_mask = ground_ocr_mask.unsqueeze(-1).expand(bsize, -1, 4) 
        ground_ocr_box = torch.masked_select(ocr_box, ground_mask.bool()).view(bsize, -1, 4)  # [bs, ocr_topk, 4]
        ground_box_mask = torch.masked_select(ocr_mask, ground_ocr_mask.bool()).view(bsize, -1)
        ground_ocr_box = ground_ocr_box * (ground_box_mask.unsqueeze(-1).expand(bsize, -1, 4))

        # final answer prediction
        # fusion_proj = torch.sum(torch.stack([out[:, 0, :] for out in out_list], dim=-1), dim=-1)
        # fusion_proj = self.vqproj(fusion_proj)

        # answer_proj = (
        #     self.get_answer_embedding(answer)
        #     if answer is not None
        #     else self.answer_embeddings
        # )
        # if question is not None and answer_proj.device != question.device:
        #     answer_proj = answer_proj.to(question.device)
        # if answer is not None:
        #     return fusion_proj, answer_proj
        # else:
        #     return fusion_proj @ answer_proj.t()

 
        # grounding evaluation
        fwd_results['ground_frame'] = ground_frame_idx
        fwd_results['ground_bbox'] = ground_ocr_box
        fwd_results['frame_topk'] = torch.tensor(self.frame_topk,device=device)
        fwd_results['ocr_topk'] = torch.tensor(self.ocr_topk,device=device)

        # # obtain the input of answer decoder: inpur = [q; ocrs; frames]  
        fwd_results['obj_mmt_in'] = frame_feat
        fwd_results['obj_mask'] = ground_frame_mask 
        fwd_results['ocr_mmt_in'] = ocr_feat  
        fwd_results['ocr_mask'] =  ground_ocr_mask
        fwd_results['txt_emb'] = question_proj


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output


class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self,
                txt_emb,
                txt_mask,
                obj_emb,
                obj_mask,
                ocr_emb,
                ocr_mask,
                fixed_ans_emb,
                prev_inds):

        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(
            dec_emb.size(0),
            dec_emb.size(1),
            dtype=torch.float32,
            device=dec_emb.device
        )

        encoder_inputs = torch.cat(
            [txt_emb, obj_emb, ocr_emb,  dec_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask, ocr_mask, dec_mask],
            dim=1
        )

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end =  txt_begin + txt_max_num 
        ocr_begin =  obj_max_num + txt_end 
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = \
            _get_causal_mask(dec_max_num, encoder_inputs.device)

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            'mmt_seq_output': mmt_seq_output,
            'mmt_txt_output': mmt_txt_output,
            'mmt_ocr_output': mmt_ocr_output,
            'mmt_dec_output': mmt_dec_output,
        }
        return results


class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = attention_mask
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        )
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=ocr_emb.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size*length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results
