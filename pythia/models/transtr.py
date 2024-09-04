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

from itertools import chain
from pythia.modules.transtr_module.multimodal_transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from pythia.modules.transtr_module.position_encoding import PositionEmbeddingSine1D
from pythia.modules.transtr_module.topk import HardtopK, PerturbedTopK
from transformers import AutoModel, AutoTokenizer


@registry.register_model("transtr")
class TRANSTR(BaseModel):
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
        self.VideoQAmodel = VideoQAmodel(self.grounding_config)

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
    
class VideoQAmodel(nn.Module):
    def __init__(self, config, text_encoder_type="roberta-base", freeze_text_encoder = False, n_query=5,
                        ocrs=4, frames=64, hard_eval=False, **kwargs):
        super(VideoQAmodel, self).__init__()
        self.d_model = 768
        encoder_dropout = 0.3
        self.mc = n_query
        self.hard_eval = hard_eval
        # text encoder
        self.text_encoder = AutoModel.from_pretrained('../../huggingface/bert-base-uncased/')
        self.tokenizer = AutoTokenizer.from_pretrained('../../huggingface/bert-base-uncased/')

        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.ocr_resize = FeatureResizer(
            input_feat_size=self.d_model,
            output_feat_size=self.d_model, 
            dropout=0.2)

        self.frame_topK, self.ocr_topK = config.frame_topk, config.ocr_topk
        self.frame_sorter = PerturbedTopK(self.frame_topK)
        self.ocr_sorter = PerturbedTopK(self.ocr_topK)

        # hierarchy 1: ocr & frame
        kwargs['num_encoder_layers'] = 2
        kwargs['num_decoder_layers'] = 2
        kwargs['nheads'] = 8 
        kwargs['d_model'] = 768
        self.ocr_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.frame_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.fo_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        
        self.vl_encoder = TransformerEncoder(TransformerEncoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.ans_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))

        # position embedding
        self.pos_encoder_1d = PositionEmbeddingSine1D()

        # cls head
        self.classifier=nn.Linear(self.d_model, 1) # ans_num+<unk>
                

    def forward(self, sample_list, fwd_results):
        """
        :param vid_frame_feat:[bs, 8, 2, 768]
        :param vid_ocr_feat:[bs, 16, 5, 2048]
        :param qns: ('what are three people sitting on?', 'what is a family having?')
        :return:
        """
        ocr_feat = fwd_results['ocr_mmt_in']
        ocr_mask = sample_list.ocr_mask
        frame_feat = fwd_results['obj_mmt_in']
        frame_mask = fwd_results['obj_mask']
        q_feat = fwd_results['txt_emb']
        q_mask = _get_mask( 
            sample_list.text_len, sample_list.text.size(1)
        )

        # Size
        B, F, D = frame_feat.size()
        O = ocr_feat.view(B, F, -1, D).size()[2]
        device = frame_feat.device
        # encode q
        # q_local, q_mask = self.forward_text(list(q_feat), device)  # [batch, q_len, d_model]
        q_local = q_feat

        #### encode v
        # frame
        frame_mask = torch.ones(B, F).bool().to(device)
        frame_local, frame_att = self.frame_decoder(frame_feat,
                                    q_local,
                                    memory_key_padding_mask=q_mask,
                                    query_pos = self.pos_encoder_1d(frame_mask , self.d_model),
                                    output_attentions=True
                                    ) # b,16,d
        
        if self.training:
            idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2) # B*16, O, topk
            # obtain the grounded frame 
            idx_frame_idx = idx_frame.max(dim=2).values # B, frame_num
            _, topk_indices = torch.topk(idx_frame_idx, self.frame_topK, dim=1) # B, frame_topk
            batch_indices = torch.arange(B).view(-1, 1).expand(-1, self.frame_topK).to(device) # B*frame_topk, 2
            frame_indices = torch.stack((batch_indices, topk_indices), dim=2).view(-1, 2)
            ground_frame = frame_indices.view(B, self.frame_topK, -1)[:, :, 1] + 1 # B,frame_topk
        else:
            idx_frame = rearrange(HardtopK(frame_att.flatten(1,2), self.frame_topK), 'b (f q) k -> b f q k', f=F).sum(-2) # B*16, O, topk
            # obtain the grounded frame 
            # idx_frame_idx = idx_frame.max(dim=2).values # B, frame_num
            # frame_indices = (idx_frame_idx == 1).nonzero(as_tuple=False) # B*frame_topk, 2
            # ground_frame = frame_indices.view(B, self.frame_topK, -1)[:, :, 1] + 1 # B,frame_topk
            idx_frame_idx = idx_frame.max(dim=2).values # B, frame_num
            _, topk_indices = torch.topk(idx_frame_idx, self.frame_topK, dim=1) # B, frame_topk
            batch_indices = torch.arange(B).view(-1, 1).expand(-1, self.frame_topK).to(device) # B*frame_topk, 2
            frame_indices = torch.stack((batch_indices, topk_indices), dim=2).view(-1, 2)
            ground_frame = frame_indices.view(B, self.frame_topK, -1)[:, :, 1] + 1 # B,frame_topk

        frame_local = (frame_local.transpose(1,2) @ idx_frame).transpose(1,2) # B, Frame_K, d)

        # obj
        reshape_ocr_feat = ocr_feat.view(B, F, O, D)
        reshape_ocr_feat = (reshape_ocr_feat.flatten(-2,-1).transpose(1,2) @ idx_frame).transpose(1,2).view(B,self.frame_topK,O,-1) # [B, topk, O, D]
        ocr_local = self.ocr_resize(reshape_ocr_feat)
        ocr_local, ocr_att = self.ocr_decoder(ocr_local.flatten(0,1),
                                            q_local.repeat_interleave(self.frame_topK, dim=0), 
                                            memory_key_padding_mask=q_mask.repeat_interleave(self.frame_topK, dim=0),
                                            output_attentions=True
                                            )  # b*16,5,d        #.view(B, F, O, -1) # b,16,5,d


        # ocr_att = ocr_att.view(B, self.frame_topK*O, -1)
        if self.training:
            idx_ocr = rearrange(self.ocr_sorter(ocr_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, ocr_topk
        else:
            idx_ocr = rearrange(HardtopK(ocr_att.flatten(1,2), self.ocr_topK), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, ocr_topk

        ocr_local = (ocr_local.transpose(1,2) @ idx_ocr).transpose(1,2).view(B, self.frame_topK, self.ocr_topK, -1) # B, frame_topK, ocr_topk, D


        # obtain the grounded ocr 
        idx_ocr = idx_ocr.view(B*self.frame_topK, O, self.ocr_topK)
        idx_ocr_idx = idx_ocr.max(dim=-1).values # B,frame_topk, O
        # idx_ocr_idx = idx_ocr_idx.view(B,-1)  # B,frame_topk*O
        ocr_indices = (idx_ocr_idx == 1).nonzero(as_tuple=False) # B*frame_topk*O,2
        if ocr_indices.shape[0] != B * self.frame_topK*self.ocr_topK:
            padding = torch.zeros(B * self.frame_topK*self.ocr_topK - ocr_indices.shape[0], 2).to(device)  # 创建大小为 [2, 2] 的全零张量
            ocr_indices = torch.cat((padding, ocr_indices), dim=0)
        
        ocr_indices = ocr_indices.view(B, self.frame_topK*self.ocr_topK, -1)[:, :, 1]  # B,frame_topk*ocr_topk
        ground_ocr = ocr_indices.view(B, self.frame_topK, self.ocr_topK) # B,frame_topk, ocr_topk
        # flatten ocr id
        flatten_ground_ocr = ((ground_frame.unsqueeze(-1)-1)*O + ground_ocr).view(B, -1).to(torch.int64) # B,frame_topk, ocr_topk
        assert (flatten_ground_ocr < 960).all()

        ocr_box = sample_list.ocr_bbox_coordinates
        ground_ocr_mask = torch.zeros((B, ocr_feat.size(1)), device=device)  # [bs, frame_num*ocr_topk]
        ground_ocr_mask.scatter_(1, flatten_ground_ocr, 1)
        ground_ocr_mask = ground_ocr_mask*ocr_mask
        new_ocr_box_mask = ground_ocr_mask.unsqueeze(-1).expand(B, -1, 4)
        ground_ocr_box = torch.zeros(B, self.frame_topK*self.ocr_topK, 4).to(device)
        
        # batch
        for i in range(B):
            # obtain ocr_box and mask in each batch
            batch_ocr_box = ocr_box[i]
            batch_mask = new_ocr_box_mask[i]
            # obtain coordinate
            selected_coords = batch_ocr_box[batch_mask.bool()].view(-1, 4)
            # padding
            num_selected = selected_coords.shape[0]
            if num_selected < self.frame_topK*self.ocr_topK:
                padded_coords = torch.cat([selected_coords, torch.zeros(self.frame_topK*self.ocr_topK - num_selected, 4).to(device)], dim=0)
            else:
                padded_coords = selected_coords[:self.frame_topK*self.ocr_topK]
            
            ground_ocr_box[i] = padded_coords

        ### hierarchy grouping
        frame_ocr = self.fo_decoder(frame_local,
                                    ocr_local.flatten(1,2),
                                    # query_pos = self.pos_encoder_1d(frame_mask.view(B,F), self.d_model), \
                                    # memory_key_padding_mask=self.win_mask.unsqueeze(0).repeat(B,1,1).to(device)
                                    ) # b,frame_topK,d
        
        ### overall fusion
        frame_feat =frame_ocr.view(B, self.frame_topK, -1)  # B, frame_topK, D
        frame_mask = torch.ones(B, self.frame_topK).bool().to(device)
        
        # grounding evaluation
        fwd_results['ground_frame'] = ground_frame
        fwd_results['ground_bbox'] = ground_ocr_box
        fwd_results['frame_topk'] = torch.tensor(self.frame_topK,device=device)
        fwd_results['ocr_topk'] = torch.tensor(self.ocr_topK,device=device)

        # # obtain the input of answer decoder: inpur = [q; ocrs; frames]  
        fwd_results['obj_mmt_in'] = frame_feat
        fwd_results['obj_mask'] = frame_mask 
        fwd_results['ocr_mmt_in'] = ocr_feat  
        fwd_results['ocr_mask'] =  ground_ocr_mask
    


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
            [obj_emb, ocr_emb,  dec_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [obj_mask, ocr_mask, dec_mask],
            dim=1
        )

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end =  txt_begin # + txt_max_num 
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
