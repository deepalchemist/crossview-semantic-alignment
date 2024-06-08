# Copyright (c) Facebook, Inc. and its affiliates.

# Last Change:  2024-05-07 14:07:21
from typing import Dict

import torch
from torch import nn, Tensor
from mmf.models.composition import NormalizationLayer
from mmf.models.rice.base_task import TaskBaseModel
from mmf.modules.losses import ContrastiveLoss

class RICEForContrastive(TaskBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.contrastive_norm = NormalizationLayer()
        self.loss_funcs = nn.ModuleDict()

        self.init_losses()

    def init_losses(self):
        self.loss_funcs["vic"] = ContrastiveLoss()
 
    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        '''video to image contrastive learning
        '''

        image_embeddings = sample_list.image # [bs d]
        image_embeddings = self.contrastive_norm(image_embeddings)  # l2 normalized and multiply a learnable factor initialized with 4
        
        video_embeddings = sample_list.video  # [bs max_frame d]
        # temporal masked pooling
        video_embeddings = video_embeddings * sample_list.video_temporal_mask.unsqueeze(-1)
        video_embeddings = torch.unbind(video_embeddings, dim=1) # split across dim=1, keepdim=False, nf * [bs d]
        video_embeddings = sum(video_embeddings)/sample_list.video_temporal_mask.sum(dim=-1, keepdim=True)  # [bs d]
        video_embeddings = self.contrastive_norm(video_embeddings)

        output_dict = {
            "scores": video_embeddings,
            "targets": image_embeddings,
        }

        loss = {}
        loss["vic_loss"] = self.loss_funcs["vic"](output_dict)
        output_dict["losses"] = loss
        return output_dict
