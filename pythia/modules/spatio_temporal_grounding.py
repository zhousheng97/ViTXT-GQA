import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionScore(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, attn_mask=None):

        attention = torch.bmm(q, k.transpose(-2, -1)).squeeze(1)
        attention = self.softmax(attention)  # bs, frame_num
        attention = attention * attn_mask
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-12)
        masked_attn = torch.where(attn_mask == 0, -10000.0, attention)

        return masked_attn 


class Temporal_Grounding_Indicator(nn.Module):
    def __init__(self, hidden_size, tau=1, is_hard=True, dropout_p=0):
        super().__init__()
        self.tau = tau
        self.is_hard = is_hard
        self.frame_pos_att = AttentionScore(hidden_size, dropout_p)
        self.frame_neg_att = AttentionScore(hidden_size, dropout_p)

    def forward(self, q_global, v_local, attn_mask, sample_list, topk):
       
        B, frame_num, D = v_local.size()[:3]
        pos_frame_score = self.frame_pos_att(q_global, v_local, attn_mask)
        neg_frame_score = self.frame_neg_att(q_global, v_local, attn_mask)
        frame_score = torch.cat((pos_frame_score.unsqueeze(1), neg_frame_score.unsqueeze(1)),1)  #[bs, 2, frame_num]

        frame_score = F.gumbel_softmax(frame_score, tau=self.tau, hard=self.is_hard, dim=1) # [bs, frame_num]  hard score
        pos_mask = frame_score[:,0,:] # [bs, frame_num]
        neg_mask = frame_score[:,1,:] # [bs, frame_num]
        assert torch.sum(pos_mask).item() + torch.sum(neg_mask).item() == B*frame_num

        pos_mask = pos_mask * attn_mask
        neg_mask = neg_mask * attn_mask

        pos_frame_score = pos_frame_score * pos_mask 
        pos_frame_score = torch.where(pos_mask == 0, -10000.0, pos_frame_score)

        _, pos_topk_indices = torch.topk(pos_frame_score, topk, dim=1, largest=True, sorted=True)
        pos_topk_mask = torch.zeros_like(pos_frame_score, device=v_local.device)
        pos_topk_mask.scatter_(1, pos_topk_indices, 1)
        assert torch.sum(pos_topk_mask).item() == B * topk

        neg_frame_score = neg_frame_score * neg_mask 
        neg_frame_score = torch.where(neg_mask == 0, -10000.0, neg_frame_score)

        _, neg_topk_indices = torch.topk(neg_frame_score, topk, dim=1, largest=False, sorted=True)
        neg_topk_mask = torch.zeros_like(neg_frame_score, device=v_local.device)
        neg_topk_mask.scatter_(1, neg_topk_indices, 1)
        assert torch.sum(neg_topk_mask).item() == B * topk

        pos_f_indice = torch.nonzero(pos_topk_mask, as_tuple=False)[:, 1].view(B, topk)
        ground_frame = torch.gather(sample_list.frame_id, 1, pos_f_indice)
        
        return ground_frame, pos_topk_mask, neg_topk_mask 


class Spatial_Grounding_Indicator(nn.Module):
    def __init__(self, hidden_size, tau=1, is_hard=True, dropout_p=0):
        super().__init__()
        self.tau = tau
        self.is_hard = is_hard
        self.ocr_pos_att = AttentionScore(hidden_size, dropout_p)
        self.ocr_neg_att = AttentionScore(hidden_size, dropout_p)

    def forward(self, q_global, v_local, v_box, v_mask, attn_mask, f_topk, o_topk, frame_num, o_frame_num):
        """
        q_global: bs,d
        v_local: bs, L, d
        """
        B, ocr_num, D = v_local.size()[:3]
        pos_ocr_score = self.ocr_pos_att(q_global, v_local, attn_mask)
        neg_ocr_score = self.ocr_neg_att(q_global, v_local, attn_mask)
        ocr_score = torch.cat((pos_ocr_score.unsqueeze(1), neg_ocr_score.unsqueeze(1)),1)  #[bs, 2, max_ocr_num]

        ocr_score = F.gumbel_softmax(ocr_score, tau=self.tau, hard=self.is_hard, dim=1) # [bs, max_ocr_num]  hard score
        pos_mask = ocr_score[:,0,:] # [bs, max_ocr_num]
        neg_mask = ocr_score[:,1,:] # [bs, max_ocr_num]
        assert torch.sum(pos_mask).item() + torch.sum(neg_mask).item() == B*ocr_num

        pos_mask = pos_mask * attn_mask
        neg_mask = neg_mask * attn_mask

        pos_ocr_score = pos_ocr_score * pos_mask
        pos_ocr_score = torch.where(pos_mask == 0, -10000.0, pos_ocr_score)
        neg_ocr_score = neg_ocr_score * neg_mask
        neg_ocr_score = torch.where(neg_mask == 0, -10000.0, neg_ocr_score)

        reshape_pos_ocr_score = pos_ocr_score.view(B, frame_num, o_frame_num)  # [bs, frame_num, ocr_frame_num]
        _, sorted_pos_indices = torch.sort(reshape_pos_ocr_score, descending=True, dim=-1)  # [bs, frame_num, ocr_frame_num]
        pos_topk_indices = sorted_pos_indices[:,:,:o_topk]  # [bs, frame_num, o_topk]
        pos_topk_mask = torch.zeros_like(reshape_pos_ocr_score, device=v_local.device)
        pos_topk_mask.scatter_(2, pos_topk_indices, 1)
        # assert torch.sum(pos_topk_mask).item() == B * o_topk * frame_num
        pos_topk_mask = pos_topk_mask.view(B, -1)

        reshape_neg_ocr_score = neg_ocr_score.view(B, frame_num, o_frame_num)  # [bs, frame_num, ocr_frame_num]
        _, sorted_neg_indices = torch.sort(reshape_neg_ocr_score, descending=False, dim=-1)  # [bs, frame_num, ocr_frame_num]
        neg_topk_indices = sorted_neg_indices[:,:,:o_topk]  # [bs, frame_num, o_topk]
        neg_topk_mask = torch.zeros_like(reshape_neg_ocr_score, device=v_local.device)
        neg_topk_mask.scatter_(2, neg_topk_indices, 1)
        # assert torch.sum(neg_topk_mask).item() == B * o_topk * frame_num
        neg_topk_mask = neg_topk_mask.view(B, -1)
        neg_topk_mask = neg_topk_mask * attn_mask

       
        # ablation: w/o TG
        # _, sorted_pos_indices = torch.sort(pos_ocr_score, descending=True, dim=-1)  # [bs, frame_num, ocr_frame_num]
        # pos_topk_indices = sorted_pos_indices[:,:o_topk]  # [bs, frame_num, o_topk]
        # pos_topk_mask = torch.zeros_like(pos_ocr_score, device=v_local.device)
        # pos_topk_mask.scatter_(1, pos_topk_indices, 1)
        # # assert torch.sum(pos_topk_mask).item() == B * o_topk * frame_num
        # pos_topk_mask = pos_topk_mask.view(B, -1)

        # _, sorted_neg_indices = torch.sort(neg_ocr_score, descending=False, dim=-1)  # [bs, frame_num, ocr_frame_num]
        # neg_topk_indices = sorted_neg_indices[:,:o_topk]  # [bs, frame_num, o_topk]
        # neg_topk_mask = torch.zeros_like(neg_ocr_score, device=v_local.device)
        # neg_topk_mask.scatter_(1, neg_topk_indices, 1)
        # # assert torch.sum(neg_topk_mask).item() == B * o_topk * frame_num
        # neg_topk_mask = neg_topk_mask.view(B, -1)
        # neg_topk_mask = neg_topk_mask * attn_mask


        # pos_topk_mask = pos_topk_mask * attn_mask 
        pos_topk_box_mask = pos_topk_mask.unsqueeze(-1).expand(B, -1, 4)
        ground_ocr_box = torch.masked_select(v_box, pos_topk_box_mask.bool()).view(B, -1, 4)  # [bs, frame_num*ocr_frame_num, 4]


        return ground_ocr_box, pos_topk_mask, neg_topk_mask 