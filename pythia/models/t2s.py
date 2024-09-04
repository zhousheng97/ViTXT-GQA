# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import math
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.layers import ClassifierLayer

from pythia.modules.spatio_temporal_grounding import  Temporal_Grounding_Indicator, Spatial_Grounding_Indicator


@registry.register_model("t2s")
class T2S(BaseModel):
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
        self.TransLayer = QTV(self.transformer_config)
        self.Grounding_Module = Grounding_Module(self.grounding_config, self.bert_config)

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
            "ref_scores": fwd_results["ref_scores"], 
            "pos_scores": fwd_results["pos_scores"], 
            "neg_scores": fwd_results["neg_scores"], 
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
        self.TransLayer(fwd_results)
        self.Grounding_Module(sample_list, fwd_results)

    def _forward_mmt(self, txt_emb, txt_mask, obj_mmt_in, obj_mask, ocr_mmt_in, ocr_mask, prev_inds ):

        mmt_results = self.mmt(
            txt_emb=txt_emb,
            txt_mask=txt_mask,
            obj_emb=obj_mmt_in,
            obj_mask=obj_mask,
            ocr_emb=ocr_mmt_in,
            ocr_mask=ocr_mask,
            prev_inds=prev_inds,
            fixed_ans_emb=self.classifier.module.weight,
        )
        return mmt_results

    def _forward_output(self, mmt_ocr_output, mmt_dec_output, ground_ocr_mask):
 
        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(
            mmt_dec_output, mmt_ocr_output, ground_ocr_mask
        )
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        return  scores

    def _forward_mmt_and_output(self, sample_list, fwd_results):
        if self.training:
            fwd_results['prev_inds'] = sample_list.train_prev_inds.clone()
            # reference
            ref_ocr_output, ref_dec_output  = self._forward_mmt(
                fwd_results['txt_emb'], fwd_results['txt_mask'], fwd_results['obj_mmt_in'], fwd_results['obj_mask'], 
                fwd_results['ocr_mmt_in'], fwd_results['ocr_mask'], fwd_results['prev_inds']
                )
            ref_score = self._forward_output(ref_ocr_output, ref_dec_output, fwd_results['ocr_mask'])
            fwd_results['ref_scores'] = ref_score

            # positive 
            pos_ocr_output, pos_dec_output  = self._forward_mmt(
                fwd_results['txt_emb'], fwd_results['txt_mask'], fwd_results['pos_obj_mmt_in'], fwd_results['pos_obj_mask'],
                fwd_results['pos_ocr_mmt_in'], fwd_results['pos_ocr_mask'], fwd_results['prev_inds']
                )
            pos_score = self._forward_output(pos_ocr_output, pos_dec_output, fwd_results['pos_ocr_mask'])
            fwd_results['pos_scores'] = pos_score

            # negative 
            neg_ocr_output, neg_dec_output = self._forward_mmt(
                fwd_results['txt_emb'], fwd_results['txt_mask'], fwd_results['neg_obj_mmt_in'], fwd_results['neg_obj_mask'], 
                fwd_results['neg_ocr_mmt_in'], fwd_results['neg_ocr_mask'], fwd_results['prev_inds']
                )
            neg_score = self._forward_output(neg_ocr_output, neg_dec_output, fwd_results['neg_ocr_mask'])
            fwd_results['neg_scores'] = neg_score

        else:
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results['prev_inds'] = torch.zeros_like(
                sample_list.train_prev_inds
            )
            fwd_results['prev_inds'][:, 0] = self.answer_processor.BOS_IDX

            # greedy decoding at test time
            for t in range(dec_step_num):
              
                # reference
                ref_ocr_output, ref_dec_output  = self._forward_mmt(
                    fwd_results['txt_emb'], fwd_results['txt_mask'], fwd_results['obj_mmt_in'], fwd_results['obj_mask'], 
                    fwd_results['ocr_mmt_in'], fwd_results['ocr_mask'], fwd_results['prev_inds']
                    )
                ref_score = self._forward_output(ref_ocr_output, ref_dec_output, fwd_results['ocr_mask'])
                fwd_results['ref_scores'] = ref_score

                # positive 
                pos_ocr_output, pos_dec_output  = self._forward_mmt(
                fwd_results['txt_emb'], fwd_results['txt_mask'], fwd_results['pos_obj_mmt_in'], fwd_results['pos_obj_mask'], 
                fwd_results['pos_ocr_mmt_in'], fwd_results['pos_ocr_mask'], fwd_results['prev_inds']
                )
                pos_score = self._forward_output(pos_ocr_output, pos_dec_output, fwd_results['pos_ocr_mask'])
                fwd_results['pos_scores'] = pos_score
                
                # negative 
                neg_ocr_output, neg_dec_output = self._forward_mmt(
                fwd_results['txt_emb'], fwd_results['txt_mask'], fwd_results['neg_obj_mmt_in'], fwd_results['neg_obj_mask'], 
                fwd_results['neg_ocr_mmt_in'], fwd_results['neg_ocr_mask'], fwd_results['prev_inds']
                )
                neg_score = self._forward_output(neg_ocr_output, neg_dec_output, fwd_results['neg_ocr_mask'])
                fwd_results['neg_scores'] = neg_score

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

class QTV(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, fwd_results):
        txt_emb = fwd_results['txt_emb']
        txt_mask = fwd_results['txt_mask']
        obj_emb = fwd_results['obj_mmt_in']
        obj_mask = fwd_results['obj_mask']
        ocr_emb = fwd_results['ocr_mmt_in']
        ocr_mask = fwd_results['ocr_mask']

        encoder_inputs = torch.cat(
            [txt_emb, obj_emb, ocr_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask, ocr_mask],
            dim=1
        )

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

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
        fwd_results['txt_emb'] = fwd_results['txt_emb'] + torch.tanh(mmt_seq_output[:, txt_begin:txt_end])
        fwd_results['obj_mmt_in'] = fwd_results['obj_mmt_in'] + torch.tanh(mmt_seq_output[:, txt_end:ocr_begin])
        fwd_results['ocr_mmt_in'] = fwd_results['ocr_mmt_in'] + torch.tanh(mmt_seq_output[:, ocr_begin:ocr_end])

class Grounding_Module(nn.Module):
    def __init__(self, grounding_config, bert_config):
        super().__init__()
        self.frame_topk = grounding_config.frame_topk
        self.ocr_topk = grounding_config.ocr_topk
        self.frame_num = grounding_config.frame_num
        self.frame_ocr_num = grounding_config.ocr_frame_num
        self.hidden_size = grounding_config.hidden_size
        self.num_hidden_layers = bert_config.num_hidden_layers
        self.tau = 1
 
        self.q_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.frame_attn = nn.Linear(self.hidden_size*2, 1)
        self.self_attn = nn.Linear(self.hidden_size, 1)

        self.frame_grounding_indicator = Temporal_Grounding_Indicator(self.hidden_size)
        self.ocr_grounding_indicator = Spatial_Grounding_Indicator(self.hidden_size)
        self.encoder = BertEncoder(bert_config)
        
    def _calculate_self_attn(self, ques, mask):
        attn = self.self_attn(ques).squeeze(-1)  # N, 20
        attn = F.softmax(attn, dim=-1)
        attn = attn * mask
        attn = attn / (attn.sum(1, keepdim=True)+ 1e-12)
        question_feature = torch.bmm(attn.unsqueeze(1), ques)  # N, 1, 768
        return question_feature

    def forward(self, sample_list, fwd_results):
    
        ocr_feat = fwd_results['ocr_mmt_in']
        ocr_mask = fwd_results['ocr_mask']
        frame_feat = fwd_results['obj_mmt_in']
        frame_mask = fwd_results['obj_mask']
        q_feat = fwd_results['txt_emb']
        q_mask = fwd_results['txt_mask']

        B, _, D = ocr_feat.size()[:3]

        q_proj = self.q_linear(q_feat)
        global_q_feat = self._calculate_self_attn(q_proj, q_mask)  # [bs, 1, 768]

        ''' stage 1. temporal grounding'''
        # obtain top-k grounded frame ids
        ground_frame, ground_frame_mask, neg_frame_mask = self.frame_grounding_indicator(
            global_q_feat, frame_feat, frame_mask, sample_list, self.frame_topk
            )
        ground_frame_mask = ground_frame_mask * frame_mask
        neg_frame_mask = neg_frame_mask * frame_mask


        ''' stage 2. spatial grounding'''
        # obtain top-k grounded ocr idx in each frame
        tensor1 = ground_frame
        tensor2 = sample_list.temporal_id # temporal id in [1, 64], 0 is padding
        ocr_box = sample_list.ocr_bbox_coordinates
        tensor1 = torch.where(tensor1 == 0, torch.tensor(1), tensor1)  # When the frame length is less than frame num (frame id=0), fill with the first frame (change frame id=1).
        equality_matrix = torch.eq(tensor2.unsqueeze(1), tensor1.unsqueeze(-1))  # [bs, frame_num, frame_num*ocr_frame_num]
        new_ocr_idx = torch.nonzero(equality_matrix, as_tuple=True)[2]
        new_ocr_idx = new_ocr_idx.view(B,-1)  # [bs, frame_num*ocr_topk] in (0, frame_num*ocr_topk-1)
        new_ocr_mask = torch.zeros((B, ocr_feat.size(1)), device=ocr_mask.device)  # [bs, frame_num*ocr_topk]
        new_ocr_mask.scatter_(1, new_ocr_idx, 1)
       
        # obtain top-k grounded ocr ids in each grounded frames
        ground_ocr_box, ground_ocr_mask, neg_ocr_mask =   self.ocr_grounding_indicator(
            global_q_feat, ocr_feat, ocr_box, ocr_mask, new_ocr_mask, self.frame_topk, self.ocr_topk, self.frame_num, self.frame_ocr_num
            )


        # grounding evaluation
        fwd_results['ground_frame'] = ground_frame
        fwd_results['ground_bbox'] = ground_ocr_box
        fwd_results['frame_topk'] = torch.tensor(self.frame_topk,device=frame_feat.device)
        fwd_results['ocr_topk'] = torch.tensor(self.ocr_topk,device=ocr_feat.device)


        # obtain the input of answer decoder: inpur = [q; ocrs; frames]  
        fwd_results['pos_obj_mmt_in'] = frame_feat 
        fwd_results['pos_obj_mask'] = ground_frame_mask 
        fwd_results['pos_ocr_mmt_in'] = ocr_feat  
        fwd_results['pos_ocr_mask'] =  ground_ocr_mask

        fwd_results['neg_obj_mmt_in'] = frame_feat 
        fwd_results['neg_obj_mask'] = neg_frame_mask 
        fwd_results['neg_ocr_mmt_in'] = ocr_feat  
        fwd_results['neg_ocr_mask'] =  neg_ocr_mask
    

class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
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

        return mmt_ocr_output, mmt_dec_output


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

def yaml_reader(yaml_path):
    file = open(yaml_path, 'r', encoding="utf-8")

    file_data = file.read()                 
    file.close()

    data = yaml.load(file_data,Loader=yaml.FullLoader)    
    return data