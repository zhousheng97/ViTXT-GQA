from transformers.activations import gelu
import torch.nn as nn
import numpy as np
import torch
import math
import copy
from transformers.modeling_outputs import BaseModelOutput
# import clip
import torch.nn.functional as F
from transformers import DistilBertConfig, BertConfig


def create_sinusoidal_embeddings(n_pos, dim, out):
    with torch.no_grad():
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads = set()

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(config.activation)
        self.activation = gelu if config.activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output
        )  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.n_layers)]
        )

    def forward(
        self,
        x,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
    ):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            if head_mask is not None:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=None,
                    output_attentions=output_attentions,
                )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_state, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Embeddings(nn.Module):
    def __init__(
        self, d_model, language_len, vision_len, dropout, sinusoidal_pos_embds
    ):
        super().__init__()
        max_position_embeddings = language_len + vision_len
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_model,
                out=self.position_embeddings.weight,
            )
        self.modality_embedding = nn.Embedding(2, d_model)
        self.language_len = language_len
        self.vision_len = vision_len
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        seq_length = embeddings.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=embeddings.device
        )  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(
            embeddings[:, :, 0]
        )  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)
        # dbg.set_trace()
        modality_embeddings = self.modality_embedding(
            torch.tensor(
                [0] * self.language_len + [1] * self.vision_len, dtype=torch.long
            ).to(embeddings.device)
        )
        embeddings = (
            embeddings + position_embeddings + modality_embeddings
        )  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class PositionEmbeddings(nn.Module):
    def __init__(
        self, d_model, max_position_embeddings, sinusoidal_pos_embds
    ):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_model,
                out=self.position_embeddings.weight,
            )
        # self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        if len(embeddings.size()) == 4:
            bsize, numf, numr, fdim = embeddings.size()
            position_ids = torch.arange(numf, dtype=torch.long, device=embeddings.device)  # (max_seq_length)
            position_ids = position_ids.view(1, -1, 1).expand(bsize, -1, numr)  # (bs, max_seq_length， num_obj)
        elif len(embeddings.size()) == 3:
            bsize, numf, fdim = embeddings.size()
            position_ids = torch.arange(numf, dtype=torch.long, device=embeddings.device)  # (max_seq_length)
            position_ids = position_ids.view(1, -1).expand(bsize, -1)  # (bs, max_seq_length， num_obj)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        return position_embeddings


class SegmentPosEmbeddings(nn.Module):
    def __init__(
        self, d_model, max_position_embeddings, sinusoidal_pos_embds
    ):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_model,
                out=self.position_embeddings.weight,
            )
        # self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        bsize, numf, fdim = embeddings.size()
        position_ids = torch.arange(numf, dtype=torch.long, device=embeddings.device)  # (max_seq_length)
        position_ids = position_ids.view(1, -1).expand(bsize, -1)  # (bs, max_seq_length， num_obj)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        return position_embeddings


class TokenTypeEmbeddings(nn.Module):
    def __init__(
        self, d_model, token_type_num
    ):
        super().__init__()
        self.modality_embedding = nn.Embedding(token_type_num, d_model)
        self.type2id = {'object': 0,
                        'segment': 1,
                        'question': 2}
        # self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, token_type):
        seq_length = embeddings.size(1)
        token_type_id = self.type2id[token_type]
        modality_embeddings = self.modality_embedding(
            torch.tensor(
                [token_type_id] * seq_length, dtype=torch.long
            ).to(embeddings.device)
        )
        return modality_embeddings


class Selector(nn.Module):
    def __init__(self, topk, selection_method='gumbel', q_dim=768, dim=768):
        super(Selector, self).__init__()
        self.linear_Q = nn.Linear(q_dim, dim)
        self.norm_Q = nn.LayerNorm(dim, eps=1e-12)

        self.linear_K = nn.Linear(dim, dim)
        self.norm_K = nn.LayerNorm(dim, eps=1e-12)

        self.topk = topk
        self.selection_method = selection_method

    @staticmethod
    def sample_gumbel(n, k):
        unif = torch.distributions.Uniform(0, 1).sample((n, k))
        g = -torch.log(-torch.log(unif))
        return g

    # @staticmethod
    def sample_gumbel_softmax(self, pi, temperature):
        n, k = pi.shape
        # dbg.set_trace()
        g = self.sample_gumbel(n, k).to(pi.device)
        h = (g + torch.log(pi)) / temperature
        h_max = h.max(dim=1, keepdim=True)[0]
        h = h - h_max
        cache = torch.exp(h)
        #     print(pi, torch.log(pi), intmdt)
        y = cache / cache.sum(dim=-1, keepdim=True)
        return y

    def forward(self, Q, K, V):  # q_feat, seg_feat, video_o
        '''
        Q: (bs, q_dim, 1)
        K: (bs, clip_num, dim), clip_num could be num_obj or num_seg
        V: (bs, clip_num, frame_num, obj_num, obj_dim)
        '''
        bs, frame_num, _ = K.shape
        obj_num, obj_dim = V.shape[-2:]
        # from IPython.core.debugger import set_trace;
        # set_trace()
        v_shape = V.shape
        # V = V.view(bs, frame_num, -1)

        # dbg.set_trace()

        Q = self.norm_Q(self.linear_Q(Q.squeeze(dim=1)))  # [bs, dim, 1] -> [bs, dim]
        K = self.norm_K(self.linear_K(K))  # [bs, numc, dim]

        logit_scale = 1
        x_logits = logit_scale * K @ Q.unsqueeze(dim=-1)
        x_logits = torch.softmax(x_logits.squeeze(dim=-1), dim=-1) # [bs, clip_num]

        selected_segs, selected_idx,  = [], []
        selected_mask = torch.zeros_like(x_logits).to(V.device)
        for _ in range(self.topk):
            # print(x_logits.shape)
            # selection_mask = self.sample_gumbel_softmax(x_logits, 1)
            selection_mask = F.gumbel_softmax(x_logits, tau=1, hard=True, dim=-1) # [bs, clip_num]
            seletion_indices = torch.nonzero(selection_mask, as_tuple=False)
            seletion_indices = seletion_indices[:, 1].view(-1, 1)
            if torch.isnan(selection_mask).sum() or torch.isinf(selection_mask).sum():
                dbg.set_trace()
            selection_mask = selection_mask.unsqueeze(dim=1) # [bs, 1, clip_num]
            if V.dim() == 3: 
                selected_segs.append(
                    torch.matmul(selection_mask, V.view(bs, frame_num,  -1)))
            else:
                selected_segs.append(
                    torch.matmul(selection_mask, V.view(bs, frame_num, -1)).view(bs, -1, obj_num, obj_dim))
            selected_idx.append(seletion_indices)
            selected_mask = torch.add(selected_mask, selection_mask.squeeze(dim=1))

        selected_segs = torch.cat(selected_segs, dim=1)  # [bs, topk * num_obj, CLIP_dim]
        selected_idx = torch.cat(selected_idx, dim=1)

        return selected_segs, selected_idx, selected_mask


class ISTA(nn.Module):
    def __init__(self, config, feature_dim, word_dim, Q, N, d_model, dropout, d_ff, h, topk=6, topj=12):
        super().__init__()
        # self.clip_topk = config.clip_topk
        self.frame_topk = config.frame_topk
        self.ocr_topk = config.ocr_topk  # max self.numr is 16
        self.frame_num = config.frame_num
        self.ocr_frame_num = config.ocr_frame_num
        self.numc = 8
        self.numf = int(32 / self.numc)
        T = self.numc + (self.ocr_topk) * self.frame_topk * self.numf
        self.position = Embeddings(d_model, Q, T, dropout, True)

        self.config = DistilBertConfig.from_pretrained(
            "/data/zsheng/huggingface/distilbert-base-uncased",
            n_layers=N,
            dim=d_model,
            dropout=dropout,
            hidden_dim=d_ff,
            attention_dropout=dropout,
            n_heads=h,
        )
        self.mmt = Transformer(self.config)
        self.seg_selector = Selector(topk=self.frame_topk)
        self.reg_selector = Selector(topk=self.ocr_topk)

        # segment embedding
        self.linear_video = nn.Linear(feature_dim, d_model)
        self.norm_video = nn.LayerNorm(d_model, eps=1e-12)

        # patch embedding
        self.linear_patch = nn.Linear(feature_dim, d_model)
        self.norm_patch = nn.LayerNorm(d_model, eps=1e-12)

        # question post bert modules
        self.linear_question = nn.Linear(word_dim, d_model)
        self.norm_question = nn.LayerNorm(d_model, eps=1e-12)

        self.d_model = d_model

        self.apply(self._init_weights)

    def get_segment_embedding(self, video):
        video = self.linear_video(video)
        video = gelu(video)
        video = self.norm_video(video)
        return video

    def get_patch_embedding(self, patch):
        patch = self.linear_patch(patch)
        patch = gelu(patch)
        patch = self.norm_patch(patch)
        return patch

    def get_question_embedding(self, question):
        question = self.linear_question(question)
        question = gelu(question)
        question = self.norm_question(question)
        return question

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

    def forward(self, config, sample_list, q_feat, q_mask, question, seg_feat, video_o):
        bsize, q_len, _ = question.shape
        seg_len = seg_feat.shape[1]
        feat_dim = seg_feat.shape[-1]

        # 1. obtain the total ocr feature on the grounded frames, choose 1 clip (4 frames) frome 16 clips
        selected_patches, ground_frame_idx, ground_frame_mask = self.seg_selector(q_feat, seg_feat, video_o)  # [bs, frame_topk, num_obj, dim]

        q_feat_tmp = q_feat.unsqueeze(dim=1).repeat(1, selected_patches.shape[1], 1, 1)  # [bs,frame_topk, num_obj, dim]
        q_feat_tmp = q_feat_tmp.view(-1, q_feat_tmp.shape[-2], q_feat_tmp.shape[-1])  # [bs * frame_topk, num_obj, dim]
        selected_patches = selected_patches.view(-1, selected_patches.shape[-2], selected_patches.shape[-1]) # [bs * frame_topk, num_obj, dim]

        # 2. obtain the grounded ocr feature
        selected_patches, ground_ocr_idx, _ = self.reg_selector(q_feat_tmp, selected_patches, selected_patches)  # [bs * frame_topk, ocr_topk, dim]
        selected_patches = selected_patches.view(bsize, -1, selected_patches.shape[-1])  # [bs, frame_topk * ocr_topk, dim]
        ground_ocr_idx = ground_ocr_idx.view(bsize, self.frame_topk, -1)  # [bs, frame_topk, ocr_topk]

        expanded_ground_frame_idx = ground_frame_idx.unsqueeze(-1)
        expanded_ground_ocr_idx = ground_ocr_idx + (expanded_ground_frame_idx)*self.ocr_frame_num
        expanded_ground_ocr_idx = expanded_ground_ocr_idx.view(bsize, -1)
        # print(torch.all((expanded_ground_ocr_idx > 0) & (expanded_ground_ocr_idx < 959)))
        ground_ocr_mask = torch.zeros(bsize, self.frame_num*self.ocr_frame_num, dtype=torch.float32).to(selected_patches.device)
        # duplicate ocr idx may appear
        ground_ocr_mask = ground_ocr_mask.scatter_(1, expanded_ground_ocr_idx, 1)
        # print(torch.sum(ground_ocr_mask, dim=1).tolist())

        # mask padding
        num_per_batch = torch.sum(ground_ocr_mask, dim=1)
        num_ones_to_add = 25 - num_per_batch
        for i in range(bsize):
            if num_ones_to_add[i] > 0:
                zero_indices = torch.where(ground_ocr_mask[i] == 0)[0]

                # 转换 num_ones_to_add[i] 为整数
                padding_num = int(num_ones_to_add[i].item())
                
                if zero_indices.size(0) >= padding_num:
                    selected_indices = zero_indices[torch.randperm(zero_indices.size(0))[:padding_num]]
                else:
                    selected_indices = zero_indices  # 如果可选位置不足，就选择所有为 0 的位置
                
                ground_ocr_mask[i, selected_indices] = 1
        

        # Position and Token Type Embedding
        question_proj = self.get_question_embedding(question)
        seg_feat = self.get_segment_embedding(seg_feat)
        patch_feat = self.get_patch_embedding(selected_patches).view(bsize, -1, self.d_model)
        
        # 3. concatenate the question, raw frames, and the grounded ocr tokens
        # vq_cat = torch.cat([question_proj, seg_feat, patch_feat], dim=1)

        # video_mask = torch.ones([bsize, seg_len + patch_feat.size(1)], dtype=torch.long, device=patch_feat.device)
        # mask = torch.cat([q_mask, video_mask], dim=1)
        # vq_cat = self.position(vq_cat)
        # attended_vq = self.mmt(x=vq_cat, attn_mask=mask)[0]

        # out_q_feat = attended_vq[:, :q_len]
        # out_seg_feat = attended_vq[:, q_len:q_len+seg_len]

        # return attended_vq, out_q_feat, out_seg_feat
        return question_proj, seg_feat, ground_frame_idx, ground_frame_mask, patch_feat, ground_ocr_mask


