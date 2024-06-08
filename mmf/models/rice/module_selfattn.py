# Copyright (c) Facebook, Inc. and its affiliates.

# Last Change:  2024-05-08 14:54:29
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch

from mmf.modules.embeddings import VideoImageEmbeddings
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.transform import prepare_4d_attention_mask

from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertConfig,
    BertPooler,
    BertPreTrainedModel,
)

from transformers import (
    CLIPProcessor, 
    CLIPPreTrainedModel,
    CLIPModel,
    CLIPConfig,
)

class SelfModel(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        # ['text_model', 'vision_model', 'visual_projection', 'text_projection']
        # text_model: ['embeddings', 'encoder', 'final_layer_norm']
        # text_model.encoder.layers is ModuleList
        clip = CLIPModel.from_pretrained(config.transformer_model_name)
        
        # Configuring
        clip_config = clip.config
        clip_config = clip_config.from_dict(OmegaConf.to_container(config, resolve=True)) # merge from local config

        config.hidden_size = clip_config.text_config.hidden_size
        self.output_attentions = clip_config.output_attentions
        self.output_hidden_states = clip_config.output_hidden_states
        self.return_dict = clip_config.return_dict
        self.visual_embedding_dim = clip_config.projection_dim
        
        # Embedding
        #self.projection = nn.Linear(self.visual_embedding_dim, config.hidden_size, bias=False)  # 512 -> 512
        self.videoimage_embeddings = VideoImageEmbeddings(config)

        # Modeling
        self.encoder = clip.text_model.encoder
        layers = nn.ModuleList([self.encoder.layers[_index] for _index in range(config.num_hidden_layers)])
        self.encoder.layers = layers
        #self.final_layer_norm = clip.text_model.final_layer_norm
        #self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=clip_config.vision_config.layer_norm_eps)
        #self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=clip_config.vision_config.layer_norm_eps)

    def forward(
        self, 
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.return_dict

        # CLIP's vision model uses layernorm before encoder
        # https://github.com/huggingface/transformers/blob/9fe3f585bb4ea29f209dc705d269fbe292e1128f/src/transformers/models/clip/modeling_clip.py#L848
        #hidden_states = self.pre_layernorm(hidden_states)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = None

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # mapping 1 to float 0, and 0 to -inf
            attention_mask = prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        #last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[:, 0, :]
        # CLIP's vision model uses layernorm for CLS output
        # https://github.com/huggingface/transformers/blob/9fe3f585bb4ea29f209dc705d269fbe292e1128f/src/transformers/models/clip/modeling_clip.py#L859
        #pooled_output = self.post_layernorm(pooled_output)


        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return dict(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
    def get_joint_embedding_video2image(
        self,
        image_embeds: torch.FloatTensor,
        video_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,  
    ) -> Dict[str, Tensor]:
        # Concat class_embedding
        # plus token_type_embedding and position embedding
        # and cat [image_embeds, video_embeds]
        hidden_states = self.videoimage_embeddings(
                image_embeds,
                video_embeds,
        )  # [bsz l d]
        
        # 
        outputs = self.forward(
            hidden_states,
            attention_mask
        )
        return outputs["pooler_output"]


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
    
    def _get_joint_embedding_video2image(
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
