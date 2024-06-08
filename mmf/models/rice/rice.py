# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
#from mmf.models.rice.classification import FashionViLForClassification
#from mmf.models.rice.composition import FashionViLForComposition
from mmf.models.rice.contrastive import RICEForContrastive
from mmf.models.rice.pretraining import RICEForPretraining
from mmf.utils.build import build_image_encoder
from mmf.utils.distributed import broadcast_tensor
from mmf.utils.general import filter_grads
from mmf.utils.modeling import get_rice_configured_parameters
from numpy.random import choice
from torch import Tensor

logger = logging.getLogger(__name__)

@registry.register_model("rice")
class RICE(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config  # w/o vocab_size, type_vocab_size
        self.training_head_type = config.training_head_type
        
        self._working_task = "task"
        self.double_view = config.get("double_view", False)
        self.freeze_image_encoder = config.get("freeze_image_encoder", False)

        if self.training_head_type == "pretraining":
            self.task_for_inference = config.task_for_inference
            self.tasks = config.tasks
            self.tasks_sample_ratio = config.get("tasks_sample_ratio", None)

    @classmethod
    def config_path(cls):
        return "configs/models/rice/defaults.yaml"
    
    @property
    def working_task(self):
        return self._working_task

    @working_task.setter
    def working_task(self, task):
        self._working_task = task

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_image_encoder:
            self.image_encoder.eval()

    def build(self):
        self.image_encoder = build_image_encoder(
            self.config.image_encoder, self.config.direct_features_input
        )
        if self.freeze_image_encoder:
            self.image_encoder = self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if self.training_head_type == "pretraining":
            self.transformer = RICEForPretraining(self.config)
        elif self.training_head_type == "contrastive":
            self.transformer = RICEForContrastive(self.config)
        else:
            raise NotImplementedError

        #if getattr(self.config, "freeze_base", False):
        #    for p in self.model.fusion.parameters():
        #        p.requires_grad = False

    def get_optimizer_parameters(self, config):
        base_lr = config.optimizer.params.lr
        weight_decay = config.optimizer.params.weight_decay
        lr_multiplier = self.config.lr_multiplier
        
        # configure image encoder
        lr_filter = []  # larger lr
        #lr_filter = lr_filter + ["patch_projection", "patch_layernorm"]
        image_encoder_params = get_rice_configured_parameters(
            self.image_encoder,
            base_lr,
            weight_decay,
            lr_filter,
            lr_multiplier,
        )
        
        for _item in image_encoder_params:
            _item["params"] = filter_grads(_item["params"])

        # configure transformer
        lr_filter = ["contrastive_norm"]  # larger lr
        #lr_filter.append("transformer.pre_layernorm")
        #lr_filter.append("transformer.post_layernorm")
        lr_filter.extend(["pooler", "embeddings", "transformer"])
        if self.training_head_type == "classification":
            lr_filter.append("classifier")
        elif self.training_head_type == "pretraining":
            lr_filter.append("heads")

        transformer_params = get_rice_configured_parameters(
            self.transformer,
            base_lr,
            weight_decay,
            lr_filter,
            lr_multiplier,
        )
        return image_encoder_params + transformer_params
    
    def extract_image_features(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # image is None when extracting query
        if hasattr(sample_list, "image"): 
            sample_list.image, sample_list.image_patch = self.image_encoder(sample_list.image)  # [bs 3 H W] to [bs d] and [bs h*w d]
            #sample_list.image_patch = sample_list.image_patch[:, 1:, :]  # FIXME: ignore CLS token
        return sample_list

    def extract_video_features(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # video is None when extracting query
        if hasattr(sample_list, "video"):
            _batch_size, _nf, d_img, h_img, w_img = sample_list.video.shape
            sample_list.video = torch.flatten(sample_list.video, start_dim=0, end_dim=1)  # include end_dim
            sample_list.video, sample_list.video_patch = self.image_encoder(sample_list.video)  # [bs*nframe 3 H W] to [bs*nframe d] and [bs*nf h*w d]
            sample_list.video = sample_list.video.view(_batch_size, -1, *sample_list.video.shape[1:]).contiguous()  # [bs nframe d]
            #sample_list.video_patch = sample_list.video_patch[:, 1:, :]  # FIXME: ignore CLS token
            sample_list.video_patch = sample_list.video_patch.view(_batch_size, -1, *sample_list.video_patch.shape[1:]).contiguous()  # [bs nf h*w d]
        return sample_list

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.training_head_type == "pretraining":
            if self.training:
                # NOTE: Sampling a task from predefined tasks
                random_idx = choice(len(self.tasks), p=self.tasks_sample_ratio)
                random_idx = broadcast_tensor(torch.tensor(random_idx).cuda())
                sample_list.task = self.tasks[random_idx.item()]
            else:
                sample_list.task = self.task_for_inference
            self.working_task = sample_list.task
        if self.training_head_type == "composition":
            sample_list.ref_image = self.image_encoder(sample_list.ref_image)
            sample_list.tar_image = self.image_encoder(sample_list.tar_image)
        else:
            # NOTE: Extracting visual features
            sample_list = self.extract_image_features(sample_list)
            sample_list = self.extract_video_features(sample_list)
        if self.training_head_type == "pretraining" and sample_list.task == "icc":
            sample_list.dv_image = self.image_encoder(sample_list.dv_image)
        return self.transformer(sample_list)
