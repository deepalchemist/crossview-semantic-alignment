# Copyright (c) Facebook, Inc. and its affiliates.

# Last Change:  2024-05-05 12:39:26
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
from mmf.modules.embeddings import BertVisioLinguisticEmbeddings
from mmf.modules.hf_layers import BertEncoderJit
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertConfig,
    BertPooler,
    BertPreTrainedModel,
)


class RICEBase(BertPreTrainedModel):
    def __init__(
        self,
        config,
        visual_embedding_dim=2048,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        config.visual_embedding_dim = visual_embedding_dim
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        self.embeddings = BertVisioLinguisticEmbeddings(config)
        self.encoder = BertEncoderJit(config)
        self.pooler = BertPooler(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.bypass_transformer = config.bypass_transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, visual_embedding_dim))

        self.init_weights()
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:

        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            if not torch.jit.is_scripting():
                extended_attention_mask = extended_attention_mask.to(
                    dtype=next(self.parameters()).dtype
                )  # fp16 compatibility
            # NOTE: 1 is to keep, 0 is to mask, this mask will be add to computed mask 
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # NOTE: Projecting visual_embedding, plus token and position embedding 
        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
        )

        # NOTE: transformer forward
        encoded_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = encoded_layers[0]
        pooled_output = self.pooler(sequence_output)
        attn_data_list = []

        if not torch.jit.is_scripting():
            if self.output_attentions:
                attn_data_list = encoded_layers[1:]
        else:
            assert (
                not self.output_attentions
            ), "output_attentions not supported in script mode"

        return sequence_output, pooled_output, attn_data_list

    def get_joint_embedding(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        return self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
            attention_mask=attention_mask,
        )
    
    def get_joint_embedding_video2image(
        self,
        video_embeddings: Tensor,
        video_embeddings_type: Tensor,
        image_embeddings: Tensor,
        image_embeddings_type: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
       
        # video_embeddings_type: [bs nf*l] no including cls token
        _b, _nf, _l, _d = video_embeddings.shape
         
        # [CLS] token embedding
        cls_token = self.cls_token.expand(_b, -1, -1)  # [bs 1 d]
        video_embeddings = torch.flatten(video_embeddings, 1, 2)  # [bs nf*l d]
        video_embeddings = torch.cat([cls_token, video_embeddings], dim=1)
        cls_embeddings_type = deepcopy(video_embeddings_type[:, 0:1])
        video_embeddings_type = torch.cat([cls_embeddings_type, video_embeddings_type], dim=1)  # [bs 1+nf*l]

        # Video embedding, project+token_type embedding+position embedding
        video_embeddings = self.embeddings.encode_image(video_embeddings, video_embeddings_type)  # [bs nf*l+1 d]
        
        # Image embedding
        image_embeddings = self.embeddings.encode_image(image_embeddings, image_embeddings_type)
 
        input_embeddings = torch.cat([video_embeddings, image_embeddings], dim=1)
        input_embeddings = self.embeddings.LayerNorm(input_embeddings)
        input_embeddings = self.embeddings.dropout(input_embeddings)
        
        # Transformer forward
        encoded_layers =  self.encoder(input_embeddings, attention_mask=None)
        sequence_output = encoded_layers[0]  # [bs l d]
        # Taking the first token hidden state  
        pooled_output = self.pooler(sequence_output)  # [bs d]
        attn_data_list = []

        if not torch.jit.is_scripting():
            if self.output_attentions:
                attn_data_list = encoded_layers[1:]
        else:
            assert (
                not self.output_attentions
            ), "output_attentions not supported in script mode"

        return sequence_output, pooled_output, attn_data_list

    def get_image_embedding(
        self,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        if self.bypass_transformer:
            return self.embeddings.projection(visual_embeddings), None, None
        else:
            return self.forward(
                visual_embeddings=visual_embeddings,
                visual_embeddings_type=visual_embeddings_type,
                attention_mask=attention_mask,
            )

    def get_text_embedding(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        return self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )


class RICEBaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_model_name = getattr(config, "bert_model_name", "bert-base-uncased")
        bert_model_dir = config.bert_model_dir 
        # NOTE: merge config and bert_config: adding vocab_size, type_vocab_size ...
        bert_config = BertConfig.from_dict(OmegaConf.to_container(config, resolve=True))

        self.config = config
        # bert_model_name download from huggingface.co
        # bert_model_dir will load from local file
        # NOTE: load pretrained weights
        self.fusion = RICEBase.from_pretrained(
            bert_model_dir,
            config=bert_config,
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
            visual_embedding_dim=config.visual_embedding_dim,
            output_attentions=config.output_attentions,
            output_hidden_states=config.output_hidden_states,
        )
        layer = nn.ModuleList([self.fusion.encoder.layer[_index] for _index in range(config.num_hidden_layers_fusion)])
        self.fusion.encoder.layer = layer

    @staticmethod
    def flatten(
        sample_list: Dict[str, Tensor],
        to_be_flattened: List[str],
        to_be_flattened_dim: List[str],
    ) -> Dict[str, Tensor]:
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])
        return sample_list

    def add_custom_params(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return sample_list

    def flatten_for_fusion(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return sample_list

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        return sample_list

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        '''
        Input:
            sample_list: a dict, collection of batch sample, key is input prefix, 
            value is batch tensor, e.g., [bs d h w]
        '''
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_fusion(sample_list)
        sample_list = self.add_post_flatten_params(sample_list)
        return self._forward(sample_list)
