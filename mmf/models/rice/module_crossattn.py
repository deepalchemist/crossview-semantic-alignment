# -*- coding: utf-8 -*-
# hokkien.ywj@gmail.com @2024-05-09 03:01:05

from __future__ import (
    absolute_import,
    division,
    print_function,
)

from typing import Any, Optional, Tuple, Union, Dict
import os
import math
import copy
import json
import logging
from copy import deepcopy
from collections import OrderedDict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mmf.utils.transform import prepare_4d_attention_mask

from transformers import CLIPModel

logger = logging.getLogger(__name__)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class CrossConfig(object):
    """Configuration class to store the configuration of a `CrossModel`.
    """
    pretrained_model_archive_map = {}
    config_name = 'cross_config.json'
    def __init__(self,
                 vocab_size_or_config_json_file=512,
                 hidden_size=512,
                 num_hidden_layers=4,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-05,
    ):
        """Constructs CrossConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CrossModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `CrossModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class CrossEmbeddings(nn.Module):
    def __init__(self, config: CrossConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.use_vision = getattr(config, "use_vision", True)
        self.use_text = getattr(config, "use_text", False)
        assert self.use_text or self.use_vision, "You should use vision or text"
        
        if self.use_text:
            self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        if self.use_vision and self.use_text:
            self.token_type_embedding = nn.Embedding(2, embed_dim)
        
        self.class_embedding = nn.Parameter(torch.randn(embed_dim))

        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )  # [1 max_position_embeddings]
        self.layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.dropout = 0.1

    def initialize_from_pretrained(self, pretrained_model):
        self.position_embedding.weight = nn.Parameter(
            deepcopy(pretrained_model.positional_embedding.weight.data), requires_grad=True
        )
        if self.use_text:
            self.token_embedding.weight = nn.Parameter(
                deepcopy(pretrained_model.token_embedding.weight.data), requires_grad=True
            )
        if self.use_text and self.use_vision:
            self.token_type_embedding.weight = nn.Parameter(
                deepcopy(pretrained_model.token_type_embedding.weight.data), requires_grad=True
            )
        logger.info("Successfully load pretrained weights for cross embeddings")
        return

    def encode_text(
        self, input_ids: Tensor,
    ) -> Tensor:
        bsz, l = input_ids.shape
        embeddings = self.token_embedding(input_ids)

        if self.use_vision:
            token_type_ids = torch.ones(
                    (bsz, l),
                    device=input_ids.device
            ).long()
            token_type_embeds = self.token_type_embedding(token_type_ids)  # [bsz l d]
            embeddings = embeddings + token_type_embeds
        return embeddings

    def encode_image(
        self,
        visual_embeds: Tensor,
    ) -> Tensor:
        bsz = visual_embeds.size(0)
        class_embeds = self.class_embedding.expand(bsz, 1, -1)
        embeddings = torch.cat([class_embeds, visual_embeds], dim=1)
        bsz, l, _ = embeddings.shape
        
        if self.use_text:
            token_type_ids = torch.zeros(
                    (bsz, l),
                    device=visual_embeds.device
            ).long()
            token_type_embeds = self.token_type_embedding(token_type_ids)  # [bsz l d]
            embeddings = embeddings + token_type_embeds
        return embeddings

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        visual_embeds: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
    ) -> Tensor:
        """
        input_ids = [batch_size, sequence_length]
        visual_embeds = [batch_size, image_feature_length, image_feature_dim]
        image_text_alignment = [batch_size, image_feature_length, alignment_dim]
        """

        # text embeddings
        if self.use_text:
            assert input_ids is not None
            text_embeds = self.encode_text(input_ids)

        # visual embeddings
        if self.use_vision:
            assert visual_embeds is not None
            visual_embeds = self.encode_image(
                visual_embeds,
            )

        if self.use_vision:
            if self.use_text:
                embeddings = torch.cat(
                    (visual_embeds, text_embeds), dim=1
                )  # concate the two
            else:
                embeddings = visual_embeds
        else:
            embeddings = text_embeds

        # Position embedding
        seq_length = embeddings.size(1)
        position_ids = self.position_ids[:, :seq_length]  # [1 seq_length]
        position_embeddings = self.position_embedding(position_ids)  # [1 seq_length d]
        embeddings = embeddings + position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = nn.functional.dropout(embeddings, p=self.dropout, training=self.training)
        return embeddings
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CrossPooler(nn.Module):
    def __init__(self, config: CrossConfig):
        super(CrossPooler, self).__init__()
        embed_dim = config.hidden_size
        
        self.layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.dense = nn.Linear(embed_dim, embed_dim, bias=False)
        #self.dense = nn.Linear(embed_dim, 1, bias=False)
        #self.activation_fn = QuickGELU()
        self.activation_fn = nn.ReLU(inplace=True)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        bsz, l, d = hidden_states.shape

        hidden_states = self.layer_norm(hidden_states)
        pooled_output = hidden_states[:, 0, :]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation_fn(pooled_output)
        return pooled_output

class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self, 
        config
    ):
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
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_probs_dropout_prob

        #self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        #self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        #self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

        in_features = self.embed_dim
        out_features = self.embed_dim * 3
        self.in_proj_weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.in_proj_bias = nn.Parameter(torch.empty(out_features))

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
        if self.in_proj_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.in_proj_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.in_proj_bias, -bound, bound)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        ''' Split across embedding dim
        '''
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if len(hidden_states) == 1:
            q_states = hidden_states[0] 
            k_states = hidden_states[0] 
            v_states = hidden_states[0] 
        else:
            assert len(hidden_states) == 3, f"Input states number should be 1 or 3, but got {len(hidden_states)}"
            q_states = hidden_states[0] 
            k_states = hidden_states[1] 
            v_states = hidden_states[2] 
        bsz, tgt_len, embed_dim = q_states.size()

        #q_states = self.q_proj(q_states) * self.scale
        #k_states = self.k_proj(k_states)
        #v_states = self.v_proj(v_states)
        in_proj_weight = torch.split(self.in_proj_weight, self.embed_dim, dim=0)
        in_proj_bias = torch.split(self.in_proj_bias, self.embed_dim, dim=0)
        q_states = F.linear(q_states, in_proj_weight[0], in_proj_bias[0])
        k_states = F.linear(k_states, in_proj_weight[1], in_proj_bias[1])
        v_states = F.linear(v_states, in_proj_weight[2], in_proj_bias[2])

        # Avoid gradient overflow
        q_states = q_states * self.scale

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(q_states, tgt_len, bsz).view(*proj_shape)
        key_states = self._shape(k_states, -1, bsz).view(*proj_shape)
        value_states = self._shape(v_states, -1, bsz).view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # [bsz * self.num_heads tgt_len src_len]

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Causal attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
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


        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # FIXME: if return softmaxed attn

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        #attn_weights = nn.functional.softmax(attn_weights, dim=-1)

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
        # attn_output: [bsz tgt_len d]
        # attn_weights_reshaped : [bsz num_heads tgt_len src_len]
        return attn_output, attn_weights_reshaped / self.scale

class TorchTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size

        self.attn = nn.MultiheadAttention(
            self.hidden_size, 
            self.num_attention_heads,
            batch_first=True,
        )
        self.ln_1 = LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(self.hidden_size, self.intermediate_size)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(self.intermediate_size, self.hidden_size))
        ]))
        self.ln_2 = LayerNorm(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        
        q_states = hidden_states
        ln_x = self.ln_1(q_states)
        '''
        query: (bsz l d)
        key and value: (bsz s d)
        attn_mask: (bsz 1 l s)
            只用来限定对k的attention范围，不管q，
            因为q如果有padding，会在输出时用q的mask过滤。
        key_padding_mask: (bsz s), 将会被扩展为 (bsz 1 l s)
        key_padding_mask和attn_mask区别：
            最终与attention_scores相加的attn_mask是
            attn_mask = key_padding_mask if attn_mask is None 
                else attn_mask + key_padding_mask
            1) self-attention. 只需要提供 key_padding_mask
            2）cross-attention. if attn_mask is None，表示q全部
                参与attention，只需要key_padding_mask. 
                如果attn_mask is not None，表示k不能全部可见，如翻译的时候。
        '''
        pooled_output, attention_weights = self.attn(
            ln_x, ln_x, ln_x, 
            key_padding_mask=attention_mask, 
            attn_mask=None, 
            need_weights=output_attentions
        )
        q_states = q_states + pooled_output
        q_states = q_states + self.mlp(self.ln_2(q_states))
        return q_states, attention_weights

class TransformerLayer(nn.Module):
    def __init__(self, config: CrossConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.attn = MultiHeadAttention(config)
        self.ln_1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(config.hidden_size, config.intermediate_size)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(config.intermediate_size, config.hidden_size))
        ]))
        self.ln_2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.ln_1(hidden_states)
        hidden_states, attn_weights = self.attn(
            hidden_states=(hidden_states, ),
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    
class CrossTransformerLayer(TransformerLayer):
    def __init__(self, config: CrossConfig):
        super(CrossTransformerLayer, self).__init__(config)
        
    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            query_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        assert len(hidden_states) == 3
        query_states = hidden_states[0]
        key_states = hidden_states[1]
        value_states = hidden_states[2]

        residual = query_states

        query_states = self.ln_1(query_states)
        query_states, attn_weights = self.attn(
            hidden_states=(query_states, key_states, value_states),
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        query_states = residual + query_states

        residual = query_states
        query_states = self.ln_2(query_states)
        query_states = self.mlp(query_states)
        query_states = residual + query_states

        outputs = (query_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    
class Transformer(nn.Module):
    def __init__(self, config: CrossConfig):
        super().__init__()
        self.config = config

        module_list = list()
        for _ in range(config.num_hidden_layers):
            module_list.append(TorchTransformerLayer(config))
            #module_list.append(CrossTransformerLayer(config))
            module_list.append(CrossTransformerLayer(config))
        self.layers = nn.ModuleList(
            module_list
        )

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = query_states
        for _index, encoder_layer in enumerate(self.layers):
            is_cross = encoder_layer.__class__.__name__ == "CrossTransformerLayer"
            
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if is_cross:
                input_states = (hidden_states, key_states, value_states)
                attention_mask = cross_attention_mask
            else:
                input_states = hidden_states
                attention_mask = self_attention_mask

            layer_outputs = encoder_layer(
                input_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return dict(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class CrossModel(nn.Module):
    def __init__(self, config: CrossConfig):
        super(CrossModel, self).__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CrossEmbeddings(config)
        self.transformer = Transformer(
             config,
        )
        self.pooler = CrossPooler(config)

        # Initializing
        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        return
   
    def initialize_from_hfmodel(self, config):
        clip = CLIPModel.from_pretrained(config.transformer_model_name)
        self.embeddings.initialize_from_pretrained(clip.text_model.embeddings)
        encoder = clip.text_model.encoder
        layers = nn.ModuleList(
            [encoder.layers[_index] for _index in range(self.config.num_hidden_layers*2)]
        )
        encoder.layers = layers
        pretrained_state_dict = encoder.state_dict()
        self.transformer.load_state_dict(pretrained_state_dict, strict=True)
        logger.info("Successfully load pretrained weights for cross transformer")
        return

    def initialize_from_openaimodel(self, config):
        from mmf.models.clip import build_model
        
        ckpt_path = config.pretrained_model_name
        try:
        # loading JIT archive
            model = torch.jit.load(ckpt_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        clip = build_model(state_dict)
        
        # Embedding
        self.embeddings.position_embedding.weight.data.copy_(
                clip.visual.positional_embedding.data
        )
        # Transformer
        pretrained_model = clip.visual.transformer  # vision model
        #pretrained_model = clip.transformer  # text model
        layers = nn.ModuleList(
            [pretrained_model.resblocks[_index] for _index in range(-self.config.num_hidden_layers*2, 0)]
        )
        state_dict = layers.state_dict()
        self.transformer.layers.load_state_dict(state_dict, strict=True)
        logger.info("Successfully load pretrained weights for cross transformer")
        return
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        query_embeds: Optional[torch.Tensor] = None,
        key_embeds: Optional[torch.Tensor] = None,
        value_embeds: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.use_text:
            assert input_ids is not None, "You have to specify input_ids"
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        
        hidden_states = query_embeds
        #hidden_states = self.embeddings(  # FIXME: cat CLS, plus position
        #    input_ids=input_ids, 
        #    visual_embeds=query_embeds,
        #)
        
        # FIXME: key and value add position embeds
        #num_frames = 10
        #seq_length = 49
        #position_ids = self.embeddings.position_ids[:, 1:]  # [1 l]
        #position_embeddings = self.embeddings.position_embedding(position_ids)  # [1 l d]
        #position_embeddings = position_embeddings.expand(num_frames, -1, -1)  # [nf l d]
        #position_embeddings = torch.flatten(position_embeddings, 0, 1).unsqueeze(0)  # [1 nf*l d]
        #key_embeds = key_embeds + position_embeddings
        #value_embeds = value_embeds + position_embeddings
 
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = None

        # expand attention_mask
        #if self_attention_mask is not None:  # FIXME: BUG
        #    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #    self_attention_mask = prepare_4d_attention_mask(
        #        self_attention_mask, 
        #        hidden_states.dtype
        #    )

        encoder_outputs = self.transformer(
            query_states=hidden_states,
            key_states=key_embeds,
            value_states=value_embeds,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0] if not return_dict else encoder_outputs["last_hidden_state"]  # [bsz l d]
        # last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = self.pooler(last_hidden_state)  # [bsz d]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return dict(
            last_hidden_state=last_hidden_state,
            pooled_output=pooled_output,
            hidden_states=encoder_outputs["hidden_states"],
            attentions=encoder_outputs["attentions"],
        )

if __name__=="__main__":
    config = CrossConfig()
    config = config.from_dict(
        {
            "use_text": False,
            "use_vision": True
        }
    )
    model = CrossModel(config)

    # Inputs
    input_ids = None
    q = torch.randn(2, 10, 128)
    k = torch.randn(2, 20, 128)
    v = k 
    bsz = q.size(0)
    tgt_len = q.size(1)
    src_len = k.size(1)
    self_attention_mask = torch.ones(
        (bsz, tgt_len),
        dtype=torch.long()
    )
    cross_attention_mask = torch.ones(
        (bsz, 1, tgt_len, src_len),
        dtype=torch.long()
    )

    # Forward
    output = model(
        input_ids,
        q,
        k,
        v,
        self_attention_mask,
        cross_attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    print(output.keys())
