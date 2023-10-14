import math
import yaml
import torch
import torch.nn as nn

from transformers import CLIPModel
from torch.autograd import Variable
from transformers.activations import ACT2FN
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling


class M_YOLO(nn.Module):
    def __init__(self, config, len_cls_map, point_cls_map):
        super(M_YOLO, self).__init__()
        with open(config, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.len_cls_map = len(len_cls_map)
        self.point_cls_map = len(point_cls_map)
        # video
        self.visual_config = CLIPVisionConfig()
        self.FFN_1 = FeedForwardLayer(self.visual_config)
        self.FFN_2 = FeedForwardLayer(self.visual_config)
        self.conv1 = nn.Conv1d(in_channels=self.visual_config.hidden_size, out_channels=self.visual_config.hidden_size,
                               kernel_size=self.config["DATASET"]["BLOCK_SIZE"],
                               stride=self.config["DATASET"]["BLOCK_SIZE"])
        self.conv2 = nn.Conv2d(in_channels=self.visual_config.hidden_size, out_channels=self.visual_config.hidden_size,
                               kernel_size=1, stride=1)
        self.attention = [PureSelfAttention(self.visual_config)
                          for _ in range(self.config["MODEL"]["TEXT_ATTN_LAYER"])]

        # text
        self.text_config = CLIPTextConfig()
        self.text_model = CLIPTextTransformer(self.text_config)
        self.text_fc = nn.Linear(self.text_config.hidden_size, self.visual_config.hidden_size)

        # fusion
        self.t_fc = nn.Linear(self.visual_config.hidden_size, self.visual_config.hidden_size)
        self.cro_attention = nn.ModuleList([PureCrossAttention(self.visual_config)
                                    for _ in range(self.config["MODEL"]["CRO_ATTN_LAYER"])])
        self.contrastive_attn = PureCrossAttention(self.visual_config, if_reverse=True)
        self.con_loss = ContrastiveLoss(alpha=self.config["MODEL"]["CON_ALPHA"], beta=self.config["MODEL"]["CON_BETA"],
                                        margin=self.config["MODEL"]["MARGIN"])

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Linear(self.visual_config.hidden_size, self.visual_config.hidden_size),
            nn.Conv2d(in_channels=1, out_channels=self.config["LOSS"]["NA"], kernel_size=1)
        )

        # classification
        # 1 represents score / self_cls_map + self.point_cls_map represents classification num of the integer part
        # 2 represents the fractional part of the number
        self.cls = nn.Linear(self.visual_config.hidden_size, 1 + self.len_cls_map + self.point_cls_map + 2)

        self._load_parameter()

    def forward(self, text, text_mask, video, video_mask):
        # TODO calculate video_block_mask --FIXED

        # Video: CLIP(freeze) -> 2 * FFN layer -> 2 * conv(resize as block) -> selfAttn -> (bs, num_block, hidden) ->
        # unsqueeze && expand(expand as anchor num) -> (bs, 1, num_block, hidden) -> conv -> (bs, 5, num_block, hidden)
        #  -> FC Layer(predict layer)
        vision_out = self.FFN_2(self.FFN_1(video))
        vision_out = vision_out.permute(0, 2, 1)
        vision_out = self.conv2(self.conv1(vision_out))
        vision_out = vision_out.permute(0, 2, 1)

        video_mask = _expand_mask(video_mask, vision_out.dtype)
        for idx, _ in enumerate(self.attention):
            vision_out = self.attention[idx](vision_out, video_mask)

        # Text: CLIP + FC Layer
        text_out = self.text_model(input_ids=text, attention_mask=text_mask)
        text_hidden = text_out.last_hidden_state
        text_hidden = self.text_fc(text_hidden)

        # Cross Attention
        cro_mask = _expand_cro_mask(video_mask, text_mask, vision_out.dtype)

        for idx, _ in enumerate(self.cro_attention):
            vision_out = self.cro_attention[idx](vision_out, text_hidden, cro_mask)

        # cross attn for ccr && ccs
        new_cro_mask = _expand_cro_mask(text_mask, video_mask, vision_out.dtype)
        new_text_out = self.t_fc(text_out)

        attn_out, rev_out = self.constrive_attn(new_text_out, vision_out, new_cro_mask)

        visual_rep = torch.stack([attn_out, rev_out], dim=2)
        # visual_rep.shape: (bsz, text_len, 2, 768)  new_text_out.shape: (bsz, text_len, 1, 768)
        cl_loss = self.contrastive_attn(nn.functional.normalize(visual_rep),
                                        nn.functional.normalize(new_text_out.unsqueeze(2)))

        vision_out = vision_out.unsqueeze(1)
        vision_out = self.final_conv(vision_out)
        output = self.cls(vision_out)

        return output, cl_loss

    def _load_parameter(self):
        # for CLIPTextTransformer
        def get_model_weight(vit_name="CLIP_model"):
            clip_model = CLIPModel.from_pretrained(vit_name)
            clip_model = clip_model.text_model
            return clip_model

        clip_model = get_model_weight()
        load_weights_dict = {k: v for k, v in clip_model.statict().items()
                             if clip_model.state_dict()[k].numel() == v.numel()}
        self.text_model.load_state_dict(load_weights_dict)


class ContrastiveLoss(nn.Module):
    def __init__(self, alpha, beta, margin=0.2, measure='cosine', max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.measure = measure
        self.max_violation = max_violation

    def forward(self, img_rep, txt_rep):
        """
            image_rep: (bs, 50, 768) -> attention weighted && reverse attention-> (bs, 4, 2, 768)
            label_rep: (bs, 4, 768) -> (bs, 4, 1, 768)
            where dim = -2 can be regarded as batch size
        """
        if self.measure == 'cosine':
            # shape: (bs, 4, 2)
            # CCR Part
            scores = self.cosine_sim_v1(img_rep, txt_rep).squeeze(-1)
            # scores[0] representation positive result
            cost_ccr = (self.margin + scores - scores[:, :, 0].unsqueeze(-1)).clamp(0)

            # CCR mask
            mask = torch.tensor([1., 0.]).unsqueeze(0).unsqueeze(1).expand_as(scores) == 1.
            I = Variable(mask)
            if torch.cuda.is_available():
                I = I.cuda()
            cost_ccr = cost_ccr.masked_fill_(I, 0)

            # shape: (bs, 4, 4)
            # CCS Part
            scores = self.cosine_sim_v2(img_rep, txt_rep)
            diagonal = torch.diagonal(scores, dim1=-2, dim2=-1).view(scores.size(0), -1, 1)
            d = diagonal.expand_as(scores)
            cost_ccs = (self.margin + scores - d).clamp(min=0)

            # CCS mask
            mask = torch.eye(scores.size(-1)).expand_as(scores) > .5
            I = Variable(mask)
            if torch.cuda.is_available():
                I = I.cuda()
            cost_ccs = cost_ccs.masked_fill_(I, 0)

            if self.max_violation:
                cost_ccs = cost_ccs.max(-1)[0]
            return self.alpha * cost_ccr.sum() + self.beta * cost_ccs.sum()

    @staticmethod
    def cosine_sim_v1(img_rep, txt_rep):
        return torch.matmul(img_rep, txt_rep.transpose(-1, -2).contiguous()) / math.sqrt(img_rep.size(-1))

    @staticmethod
    def cosine_sim_v2(img_rep, txt_rep):
        img_rep = img_rep[:, :, 0, :]
        txt_rep = txt_rep.squeeze()
        return torch.matmul(img_rep, txt_rep.transpose(-1, -2).contiguous()) / math.sqrt(img_rep.size(-1))


class CLIPTextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embeded_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embeded_dim, eps=config.layer_norm_eps)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # FIXME: confusion point: why causal mask use in clip training
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1)
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidde_states,
            attentions=encoder_outputs.attentions
        )


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, position_ids, inputs_embeds):
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embedding = inputs_embeds + position_embeddings

        return embedding


class CLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                input_embeds,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hiddens
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = input_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_output = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions
            )

            hidden_states = layer_output[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_output[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions])
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask, causal_attention_mask, output_attentions=False):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.hidden_size
        self.lay_norm = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.lay_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class PureSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask):
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class PureCrossAttention(nn.Module):
    def __init__(self, config, if_reverse=False):
        super().__init__()
        self.config = config
        self.if_reverse = if_reverse
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        if self.if_reverse:
            self.rev_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, q, kv, attention_mask):
        bsz, q_len, embed_dim = q.size()
        _, kv_len, _ = kv.size()

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(kv), -1, bsz)
        value_states = self._shape(self.v_proj(kv), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, q_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, q_len, kv_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, q_len, kv_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, q_len, kv_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        if self.if_reverse:
            rev_attn = nn.functional.softmax(1 - attn_probs, dim=-1)
            rev_output = torch.bmm(rev_attn, value_states)
            rev_output = rev_output.view(bsz, self.num_heads, q_len, self.head_dim)
            rev_output = rev_output.transpose(1, 2)
            rev_output = rev_output.reshape(bsz, q_len, embed_dim)
            rev_output = self.rev_proj(rev_output)
            return attn_output, rev_output

        return attn_output


def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _expand_cro_mask(mask1, mask2, dtype):
    bsz, src_len1 = mask1.size()
    _, src_len2 = mask2.size()

    mask = torch.bmm(mask1.unsqueeze(-1), mask2.unsqueeze(1))

    expanded_mask = mask[:, None, :, :].expand(bsz, 1, src_len1, src_len2).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
