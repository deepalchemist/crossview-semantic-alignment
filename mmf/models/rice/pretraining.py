# Copyright (c) Facebook, Inc. and its affiliates.

import os
import logging
from copy import deepcopy
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmf.models.rice.base_task import TaskBaseModel
from mmf.models.rice.module_selfattn import SelfModel
from mmf.models.rice.module_crossattn import (
        CrossConfig, 
        CrossModel,
        QuickGELU
)
from mmf.modules.losses import (
    ContrastiveLoss,
    CrossEntropyLoss,
    HardNegativeCrossEntropyLoss,
    MSELoss,
    SupervisedContrastiveLoss,
    SoftLabelCrossEntropyLoss,
)
from mmf.models.composition import NormalizationLayer
from mmf.modules.ot import optimal_transport_dist
from mmf.utils.build import build_image_encoder
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.transform import prepare_4d_attention_mask
from mmf.utils.distributed import gather_tensor_along_batch_with_backward, get_rank

from transformers.modeling_bert import (
    BertOnlyNSPHead,
    BertForPreTraining,
    BertPredictionHeadTransform,
)

logger = logging.getLogger(__name__)

class CosSim(nn.Module):
    def __init__(self, nfeat, nclass):
        super().__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.projection = nn.Parameter(torch.randn(nfeat, nclass), requires_grad=True)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        projection_norm = F.normalize(self.projection, p=2, dim=0)
        logits = torch.matmul(x, projection_norm)
        return logits


class RICEForPretraining(TaskBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.task_for_inference = config.task_for_inference
        self.tasks = config.tasks
        self.double_view = config.get("double_view", False)
        self.no_sharing = config.get("no_sharing", False)
        
        # Self-attention model
        #self.transformer = SelfModel(config)

        # Cross-attention  model
        cross_config = CrossConfig()
        cross_config = cross_config.from_dict(config.transformer_model)
        self.transformer = CrossModel(cross_config)

        #if self.no_sharing:
        #    self.transformer_2 = deepcopy(self.transformer)
        
        self.heads = nn.ModuleDict()
        self.loss_funcs = nn.ModuleDict()

#        if "mpfc" in config.tasks and config.bypass_transformer:
#            self.image_tokenizer = build_image_encoder(
#                config.image_tokenizer,
#                config.direct_features_input,
#            )
#            self.image_tokenizer = self.image_tokenizer.eval()
#            for param in self.image_tokenizer.parameters():
#                param.requires_grad = False
#
        self.build_heads()
        self.build_losses()
        #self.heads.apply(self.initialize_head_weights)
        self.transformer.initialize_from_openaimodel(cross_config)
        return

#    def get_image_embedding(self, *args, **kwargs):
#        if self.no_sharing:
#            return self.transformer_2.get_image_embedding(*args, **kwargs)
#        else:
#            return self.fusion.get_image_embedding(*args, **kwargs)
#
#    def get_text_embedding(self, *args, **kwargs):
#        if self.no_sharing:
#            return self.fusion_2.get_text_embedding(*args, **kwargs)
#        else:
#            return self.fusion.get_text_embedding(*args, **kwargs)
#
#    def get_joint_embedding(self, *args, **kwargs):
#        return self.fusion.get_joint_embedding(*args, **kwargs)

    def build_heads(self):
        if "vim" in self.tasks:
            input_dim = self.transformer.config.hidden_size
            #self.heads["vim"] = nn.Sequential(
            #    nn.Linear(input_dim, input_dim * 2, bias=True),
            #    nn.LayerNorm(input_dim * 2),
            #    nn.GELU(),
            #    nn.Linear(input_dim * 2, 1, bias=False)
            #)
            self.heads["vim"] = nn.Linear(input_dim, 1, bias=False)
            #self.heads["vim"] = nn.Identity()

        if "rec" in self.tasks:
            num_attention_heads = self.transformer.config.num_attention_heads
            num_hidden_layers = self.transformer.config.num_hidden_layers
            input_dim = num_hidden_layers * num_attention_heads
            self.heads["rec"] = nn.Sequential(
                    nn.Conv2d(input_dim, input_dim//2, kernel_size=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(input_dim//2, 1, kernel_size=(1, 1)),
                    nn.ReLU(inplace=True)
            )
            self.heads["rec"].apply(self.initialize_head_weights)
        if "mlm" in self.tasks:
            bert_masked_lm = BertForPreTraining.from_pretrained(
                #self.config.bert_model_name,  # TODO(fixed)
                self.config.bert_model_dir,
                config=self.bert.config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
            )
            self.heads["mlm"] = deepcopy(bert_masked_lm.cls.predictions)
            self.fusion._tie_or_clone_weights(
                self.heads["mlm"].decoder, self.fusionembeddings.word_embeddings
            )
        if "mpfr" in self.tasks:
            self.heads["mpfr"] = nn.Sequential(
                BertPredictionHeadTransform(self.fusion.config),
                nn.Linear(
                    self.fusion.config.hidden_size,
                    self.config.visual_embedding_dim,
                    bias=False,
                ),
            )
            self.fusion._tie_or_clone_weights(
                self.heads["mpfr"][1], self.fusion.embeddings.projection
            )
            self.heads["mpfr"][1].weight = nn.Parameter(
                self.heads["mpfr"][1].weight.t()
            )
            self.heads["mpfr"][1].bias = nn.Parameter(
                torch.zeros(self.config.visual_embedding_dim)
            )
        if "mpfc" in self.tasks:
            self.heads["mpfc"] = nn.Sequential(
                BertPredictionHeadTransform(self.fusion.config),
                nn.Linear(
                    self.fusion.config.hidden_size,
                    1024,
                ),
            )
        if "pac" in self.tasks:
            self.heads["pac"] = CosSim(self.fusion.config.hidden_size, 2232)
        return

    def initialize_head_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        return

    def build_losses(self):
        self.loss_funcs["vic"] = ContrastiveLoss()
        if "vim" in self.tasks:
            #self.loss_funcs["vim"] = CrossEntropyLoss()
            self.loss_funcs["vim"] = HardNegativeCrossEntropyLoss()
        if "rec" in self.tasks:
            self.loss_funcs["rec"] = MSELoss()
            #self.loss_funcs["rec"] = ContrastiveLoss()
        if "itm" in self.tasks:
            self.loss_funcs["itm"] = CrossEntropyLoss()
        if "mlm" in self.tasks:
            self.loss_funcs["mlm"] = CrossEntropyLoss(ignore_index=-1)
        if "mpfr" in self.tasks:
            self.loss_funcs["mpfr"] = MSELoss()
        if "mpfc" in self.tasks:
            self.loss_funcs["mpfc"] = CrossEntropyLoss()
        if "mvc" in self.tasks:
            self.loss_funcs["mvc"] = SupervisedContrastiveLoss()
        if "pac" in self.tasks:
            self.loss_funcs["pac"] = SoftLabelCrossEntropyLoss()
        if "icc" in self.tasks:
            self.loss_funcs["icc"] = ContrastiveLoss()
    
    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # NOTE: sequence attention mask of transformer, 1 is keep, 0 is masked
        bsz, l, _ = sample_list.image_patch.shape
        _, nf, vl, d = sample_list.video_patch.shape
        device = sample_list.image_patch.device
        tgt_len = l  # FIXME: include cross model's CLS token
        src_len = [nf, vl]  # FIXME

        image_mask = torch.ones((bsz, tgt_len), device=device).long()  # including CLS
        #sample_list.self_attention_mask = image_mask  # FIXME: BUG
        sample_list.self_attention_mask = (1 - image_mask).bool()  # PyTorch multi_head_attn, True means ignore

        video_mask = torch.ones((bsz, *src_len), device=device).long()
        video_mask = video_mask * sample_list.video_temporal_mask[:, :, None]
        cross_attention_mask = prepare_4d_attention_mask(
                video_mask.view(bsz, -1), 
                sample_list.image_patch.dtype, 
                tgt_len
        )  # [bsz 1 tgt_len src_len]
        sample_list.cross_attention_mask = cross_attention_mask
        sample_list.attention_mask = torch.cat(
            (
                image_mask,
                video_mask.view(bsz, -1)
            ),
            dim=-1,
        )  # [bsz nf*l+l+1]
        sample_list.video_patch = sample_list.video_patch.view(bsz, -1, d)  # [bsz nf*l d]

        if self.double_view and self.training and sample_list["task"] != "mvc":
            sample_list["input_ids"] = sample_list["input_ids"].repeat(2, 1)
            sample_list["segment_ids"] = sample_list["segment_ids"].repeat(2, 1)
            sample_list["input_mask"] = sample_list["input_mask"].repeat(2, 1)
            sample_list["targets"] = sample_list["targets"].repeat(2)
            if sample_list["task"] == "mlm":
                sample_list["input_ids_masked"] = sample_list[
                    "input_ids_masked"
                ].repeat(2, 1)
                sample_list["lm_label_ids"] = sample_list["lm_label_ids"].repeat(2, 1)

        if sample_list["task"] == "mvc":
            sample_list["visual_embeddings_type"] = torch.zeros(
                (b // 2, l), device=device
            ).long()
            sample_list["attention_mask"] = torch.cat(
                (
                    sample_list["input_mask"],
                    torch.ones((b // 2, l), device=device).long(),
                ),
                dim=-1,
            )
        
        if sample_list["task"] in ["itm", "mlm", "mpfr", "mpfc", "icc"]:
            # NOTE: sequence attention mask of transformer, 1 is keep, 0 is masked
            sample_list["attention_mask"] = torch.cat(
                (
                    sample_list["input_mask"],  # word mask, [bs max_word]
                    torch.ones((b, l), device=device).long(),
                ),
                dim=-1,
            )

        if sample_list["task"] in ["mpfr", "mpfc"]:
            if self.double_view:
                sample_list["image_masks"] = sample_list["image_masks"].repeat(2, 1)
            mask = sample_list["image_masks"] == 0
            mask = mask.float().unsqueeze(-1)
            sample_list["masked_image"] = sample_list["image"] * mask

        return sample_list
    
    def preprocessing_input(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        '''
        Input:
            sample_list: a dict, collection of batch sample, key is input prefix, 
            value is batch tensor, e.g., [bs d h w]
        '''
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_fusion(sample_list)
        sample_list = self.add_post_flatten_params(sample_list)
        return sample_list

    @torch.no_grad()
    def get_patch_labels(self, image, chunk_size=8):
        batch_size = image.shape[0]
        assert batch_size % chunk_size == 0
        # We need to override eval() as this image_tokenizer is a submodule
        self.image_tokenizer = self.image_tokenizer.eval()
        indices = []
        for i in range(batch_size // chunk_size):
            _, _, idx = self.image_tokenizer(
                image[i * chunk_size : (i + 1) * chunk_size]
            )
            indices.append(idx)
        indices = torch.cat(indices, dim=0)
        return indices.long()

    @torch.no_grad()
    def get_hard_pairs(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Hard negative pairs mining
        # FIXME: not support multi-gpu mining
        if self.training:
            reset_train = True
            self.eval()
        else:
            reset_train = False

        itc_dict = self._forward_itc(sample_list)
        image_embeddings = itc_dict["scores"]
        text_embeddings = itc_dict["targets"]
        correlations = image_embeddings @ text_embeddings.t()
        batch_size = correlations.shape[0]
        # under double_view mode we have more than one positives
        diag = torch.eye(batch_size).bool()
        if self.double_view:
            bs = batch_size // 2
            diag[:bs, bs:] = diag[:bs, :bs]
            diag[bs:, :bs] = diag[:bs, :bs]
        correlations[diag] = -1
        # FIXME: more complicated sampling strategy
        hard_text_index = torch.argmax(correlations, dim=1)
        combine_index = torch.arange(batch_size).to(image_embeddings.device)
        combine_index_index = torch.rand(batch_size) > 0.5
        combine_index[combine_index_index] = hard_text_index[combine_index_index]

        if reset_train:
            self.train()

        sample_list["input_ids"] = sample_list["input_ids"][combine_index]
        sample_list["segment_ids"] = sample_list["segment_ids"][combine_index]
        sample_list["input_mask"] = sample_list["input_mask"][combine_index]
        if "attention_mask" in sample_list.keys():
            sample_list["attention_mask"] = sample_list["attention_mask"][combine_index]
        sample_list["targets"][combine_index_index] = 0

        return sample_list

    #@torch.no_grad()
    def get_hard_pairs_video2image(
        self, 
        sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # Hard negative <video, image> pairs mining
        # FIXME: not support multi-gpu gather mining
        with torch.no_grad():
            if self.training:
                reset_train = True
                self.eval()
            else:
                reset_train = False

            itc_dict = self._forward_vic(sample_list)
            video_embeddings = itc_dict["scores"]
            image_embeddings = itc_dict["targets"]
            correlations = video_embeddings @ image_embeddings.t()  # [bs bs]
            batch_size = correlations.shape[0]

            diag = torch.eye(batch_size).bool()
            correlations[diag] = -1
            # FIXME: more complicated sampling strategy
            hard_text_index = torch.argmax(correlations, dim=1)
            combine_index = torch.arange(batch_size).to(image_embeddings.device)
            combine_index_index = torch.rand(batch_size) > 0.5  # random number from [0, 1), random sampling negative and keep positive for ce loss  
            combine_index[combine_index_index] = hard_text_index[combine_index_index]
            combine_index = combine_index.contiguous()

            if reset_train:
                self.train()
        
        # FIXME: who to change order?
        # image, image_id, image_masks, image_patch, attention_mask
        sample_list.image_patch_hard = sample_list.image_patch[combine_index]
        # attention_mask remain unchanged as video order unchanged,
        # all image has the same mask filled with all 1
        #if "attention_mask" in sample_list.keys():
        #    sample_list.attention_mask = sample_list.attention_mask[combine_index]

        sample_list.targets[combine_index_index] = 0  # pair target for ce loss, 0 denotes negative pair, 1 denotes positive pair 

        return sample_list

   
    def get_joint_embedding_cross_video2image(
        self, 
        image_embeds: torch.FloatTensor,  # [bs l d], EmbeddingLayer will cat CLS
        video_embeds: torch.FloatTensor,  # [bs nf*l d]
        self_attention_mask: Optional[torch.Tensor] = None,  # [bs l+1], including CLS
        cross_attention_mask: Optional[torch.Tensor] = None,  # [bs 1 l+1 nf*l]
        output_attentions: Optional[torch.Tensor] = None,
    ) -> Tensor:
        output_dict = self.transformer(
            input_ids=None,
            query_embeds=image_embeds,
            key_embeds=video_embeds,
            value_embeds=video_embeds,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )
        pooled_output = output_dict["pooled_output"]
        attentions = output_dict["attentions"]
        return pooled_output, attentions

    def _forward_itm(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.get_hard_pairs(sample_list)
        _, pooled_output, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        logits = self.heads["itm"](pooled_output)
        reshaped_logits = logits.contiguous().view(-1, 2)
        output_dict = {"scores": reshaped_logits}

        loss = {}
        loss["itm_loss"] = self.loss_funcs["itm"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_vim_v2i(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        NUM_SAMPLE = 4
        NUM_NEGATIVE = NUM_SAMPLE - 1

        # Get similarity matrix
        #if self.training:
        #    reset_train = True
        #    self.eval()
        #else:
        #    reset_train = False

        with torch.no_grad():
            #vic_dict = self._forward_vic(sample_list)
            #video_embeds = vic_dict["scores"]
            #image_embeds = vic_dict["targets"]

            video_embeds = sample_list.video_embeddings
            #video_embeds = gather_tensor_along_batch_with_backward(video_embeds)
            image_embeds = sample_list.image_embeddings
            image_embeds_allgpus = gather_tensor_along_batch_with_backward(
                image_embeds
            )
            correlations = video_embeds @ image_embeds_allgpus.t()  # [bsz g_bsz]

        #if reset_train:
        #    self.train()

        bsz = correlations.shape[0]
        # Get hard index
        video_patch = sample_list.video_patch
        #video_patch = gather_tensor_along_batch_with_backward(video_patch) 
        crossattn_mask = sample_list.cross_attention_mask  # [bsz 1 l+1 nf*l]
        #crossattn_mask = gather_tensor_along_batch_with_backward(crossattn_mask) 
        image_patch_allgpu = gather_tensor_along_batch_with_backward(
            sample_list.image_patch.contiguous()
        )  # [g_bsz l d]
        selfattn_mask =  gather_tensor_along_batch_with_backward(
            sample_list.self_attention_mask
        )  # [g_bsz l+1]
        _, nfl, _ = video_patch.size()
        g_bsz, l, hidden_size = image_patch_allgpu.size()

        assert correlations.size(0) == bsz \
            and correlations.size(1) == g_bsz
        #local_rank = int(os.environ["LOCAL_RANK"])
        labels = bsz * get_rank() + torch.arange(
            bsz, device=video_patch.device
        )
        #labels = torch.arange(bsz, device=correlations.device)
        _, indexes = correlations.sort(descending=True)
        acc = (labels == indexes[:, 0]).sum().item() / bsz * 100.
        logger.info(f"Accuracy of dual model: {acc:.2f}%")
        indexes = indexes.detach().cpu().numpy().tolist()  # [bsz g_bsz]
        hard_index_lst = list()
        for hard_i, _index in enumerate(indexes):
            tgt = labels[hard_i]
            hard_index_lst.append([t for t in _index if t!=tgt][:NUM_NEGATIVE])

        retrieve_logits_lst = []
        pairwise_target_lst = []
        retrieve_attn_weights_lst = []

        # Go through 
        STEP_SIZE = 2  # set smaller to reduce memory cost
        assert bsz % STEP_SIZE == 0
        split_size = [STEP_SIZE] * (bsz // STEP_SIZE)

        video_embed_splits = torch.split(video_patch, split_size, dim=0)
        crossattn_mask_splits = torch.split(crossattn_mask, split_size, dim=0)
        for i in range(len(split_size)):
            video_embed_step = video_embed_splits[i]  # [STEP_SIZE nfl d]
            video_embed_step = video_embed_step.unsqueeze(1).repeat(1, NUM_SAMPLE, 1, 1)
            video_embed_step = video_embed_step.view(-1, nfl, hidden_size)  # (STEP_SIZE*NUM_SAMPLE, nfl, d)
            
            crossattn_mask_step = crossattn_mask_splits[i]  # [STEP_SIZE 1 l+1 nf*l]
            crossattn_mask_step = crossattn_mask_step.unsqueeze(1).repeat(1, NUM_SAMPLE, 1, 1, 1)
            crossattn_mask_step = torch.flatten(crossattn_mask_step, 0, 1)  # [STEP_SIZE*NUM_SAMPLE 1 l+1 nf*l]

            # Hard Negative Sampling
            _tic = i * STEP_SIZE 
            _toc = (i+1) * STEP_SIZE
            _toc = _toc if _toc < bsz else bsz
            sample_lst = list()
            sample_mask_lst = list()
            for j in range(_tic, _toc):
                # Random sampling
                #index_neg = list(range(b_visual))
                #index_neg.pop(j)
                #index_neg = random.sample(index_neg, NUM_SAMPLE-1)

                # Hard sampling
                index_neg = hard_index_lst[j]
                tgt = labels[j]
                pos = image_patch_allgpu[tgt:tgt+1]  # (1 l d)
                neg = image_patch_allgpu[index_neg]  # (NUM_NEGATIVE l d)
                pair = torch.cat([pos, neg], dim=0)  # (NUM_SAMPLE l d)
                sample_lst.append(pair)

                pos_mask = selfattn_mask[tgt:tgt+1]  # [1 l+1]
                neg_mask = selfattn_mask[index_neg]  # (NUM_NEGATIVE l+1)
                pair_mask = torch.cat([pos_mask, neg_mask], dim=0)
                sample_mask_lst.append(pair_mask)

            image_embed_step = torch.stack(sample_lst, dim=0)  # (STEP, NUM_SAMPLE l d)
            image_embed_step = image_embed_step.view(-1, l, hidden_size)  # (STEP*NUM_SAMPLE l d), view=concat all steps
            selfattn_mask_step = torch.stack(sample_mask_lst, dim=0)  # (STEP, NUM_SAMPLE, l+1)
            selfattn_mask_step = torch.flatten(selfattn_mask_step, 0, 1)

            # pooled_output (STEP_SIZE*NUM_SAMPLE d)
            # attn_weights (STEP_SIZE*NUM_SAMPLE num_heads L S)
            pooled_output, attn_weights = self.get_joint_embedding_cross_video2image(
                image_embed_step,  # [_ l d], including hard negatives
                video_embed_step,  # [_ nf*l d]
                selfattn_mask_step,  # [_ l+1], including CLS
                crossattn_mask_step,  # [_ 1 l+1 nf*l]
                output_attentions=True,
            ) 
            #retrieve_logits_step = self.heads["vim"](pooled_output).squeeze(-1).view(STEP_SIZE, NUM_SAMPLE)

            retrieve_logits_step = F.normalize(pooled_output, dim=-1) @ F.normalize(self.heads["vim"].weight, dim=-1).t()
            retrieve_logits_step = retrieve_logits_step.squeeze(-1).view(STEP_SIZE, NUM_SAMPLE)

            #retrieve_logits_step = self.heads["vim"](pooled_output)  # (STEP_SIZE*NUM_SAMPLE, 2)
            pairwise_target_step = torch.zeros(
                    (STEP_SIZE, NUM_SAMPLE),
                    dtype=torch.long,
                    device=retrieve_logits_step.device
            )
            pairwise_target_step[:, 0] = 1
            attn_weights = [
                t.view(STEP_SIZE, NUM_SAMPLE, *t.shape[1:])[:,0, ...] for t in attn_weights
            ]  # only retain positive attn_weights, [STEP_SIZE num_heads l s]

            retrieve_logits_lst.append(retrieve_logits_step)
            pairwise_target_lst.append(pairwise_target_step)
            retrieve_attn_weights_lst.append(attn_weights)

        retrieve_logits = torch.cat(retrieve_logits_lst, dim=0)  # [bsz NUM_SAMPLE]
        #retrieve_weights = torch.cat(retrieve_attn_weights_lst, dim=0)
        pairwise_target = torch.cat(pairwise_target_lst, dim=0).view(-1)  # [bsz NUM_SAMPLE] -> [bsz*NUM_SAMPLE]
        print(retrieve_logits[:2,:])
        #print(pairwise_target)
        sample_list.targets = pairwise_target

        model_output = {
                "scores": retrieve_logits,
                "diag": False,
        }
        loss = {}
        loss["vim_loss"] = self.loss_funcs["vim"](sample_list, model_output)
        model_output["losses"] = loss

        #rec_loss = self._forward_rec(sample_list)["losses"]
        #model_output["losses"].update(rec_loss)

        return model_output

    def _forward_vim(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        NUM_SAMPLE = 4
        NUM_NEGATIVE = NUM_SAMPLE - 1

        with torch.no_grad():
            #_ = self._forward_vic(sample_list)
            image_embeds = sample_list.image_embeddings
            video_embeds = sample_list.video_embeddings
            video_embeds = gather_tensor_along_batch_with_backward(
                video_embeds
            )
            correlations = image_embeds @ video_embeds.t()  # [bsz g_bsz]

        # Get hard index
        image_patch = sample_list.image_patch # [bsz l d]
        selfattn_mask = sample_list.self_attention_mask  # [bsz l]

        video_patch = gather_tensor_along_batch_with_backward(
            sample_list.video_patch
        )  # [g_bsz nfl d]
        crossattn_mask = gather_tensor_along_batch_with_backward(
            sample_list.cross_attention_mask
        )  # [g_bsz 1 l nf*l]
        
        g_bsz, nfl, _ = video_patch.size()
        bsz, l, hidden_size = image_patch.size()

        assert correlations.size(0) == bsz \
            and correlations.size(1) == g_bsz
        labels = bsz * get_rank() + torch.arange(
            bsz, device=image_patch.device
        )
        #labels = torch.arange(bsz, device=correlations.device)

        _, indexes = correlations.sort(dim=-1, descending=True)
        acc = (labels == indexes[:, 0]).sum().item() / bsz * 100.
        logger.info(f"Accuracy of dual model: {acc:.2f}%")

        indexes = indexes.detach().cpu().numpy().tolist()  # [bsz g_bsz]
        
        hard_index_lst = list()
        for hard_i, _index in enumerate(indexes):
            tgt = labels[hard_i]
            hard_index_lst.append([t for t in _index if t!=tgt][:NUM_NEGATIVE])

        retrieve_logits_lst = []
        pairwise_target_lst = []
        retrieve_attn_weights_lst = []

        # Go through 
        STEP_SIZE = 2  # set smaller to reduce memory cost
        assert bsz % STEP_SIZE == 0
        split_size = [STEP_SIZE] * (bsz // STEP_SIZE)

        image_embed_splits = torch.split(image_patch, split_size, dim=0)
        selfattn_mask_splits = torch.split(selfattn_mask, split_size, dim=0)
        for i in range(len(split_size)):
            image_embed_step = image_embed_splits[i]  # [STEP_SIZE l d]
            image_embed_step = image_embed_step.unsqueeze(1).repeat(1, NUM_SAMPLE, 1, 1)
            image_embed_step = image_embed_step.view(-1, l, hidden_size)  # (STEP_SIZE*NUM_SAMPLE, l, d)
            
            selfattn_mask_step = selfattn_mask_splits[i]  # [STEP_SIZE l]
            selfattn_mask_step = selfattn_mask_step.unsqueeze(1).repeat(1, NUM_SAMPLE, 1)
            selfattn_mask_step = torch.flatten(selfattn_mask_step, 0, 1)  # [STEP_SIZE*NUM_SAMPLE l]

            # Hard Negative Sampling
            _tic = i * STEP_SIZE 
            _toc = (i+1) * STEP_SIZE
            _toc = _toc if _toc < bsz else bsz
            sample_lst = list()
            sample_mask_lst = list()
            for j in range(_tic, _toc):
                # Random sampling
                #index_neg = list(range(b_visual))
                #index_neg.pop(j)
                #index_neg = random.sample(index_neg, NUM_SAMPLE-1)

                # Hard sampling
                index_neg = hard_index_lst[j]
                tgt = labels[j]
                pos = video_patch[tgt:tgt+1]  # (1 nfl d)
                neg = video_patch[index_neg]  # (NUM_NEGATIVE nfl d)
                pair = torch.cat([pos, neg], dim=0)  # (NUM_SAMPLE nfl d)
                sample_lst.append(pair)

                pos_mask = crossattn_mask[tgt:tgt+1]  # [1 1 l nfl]
                neg_mask = crossattn_mask[index_neg]  # (NUM_NEGATIVE 1 l nfl)
                pair_mask = torch.cat([pos_mask, neg_mask], dim=0)
                sample_mask_lst.append(pair_mask)

            video_embed_step = torch.stack(sample_lst, dim=0)  # (STEP NUM_SAMPLE nfl d)
            video_embed_step = video_embed_step.view(-1, nfl, hidden_size)  # (STEP*NUM_SAMPLE nfl d), view=concat all steps
            crossattn_mask_step = torch.stack(sample_mask_lst, dim=0)  # (STEP, NUM_SAMPLE, 1 l nfl)
            crossattn_mask_step = torch.flatten(crossattn_mask_step, 0, 1)  # (STEP*NUM_SAMPLE 1 l nfl)

            # pooled_output (STEP_SIZE*NUM_SAMPLE d)
            # attn_weights (STEP_SIZE*NUM_SAMPLE num_heads L S)
            pooled_output, attn_weights = self.get_joint_embedding_cross_video2image(
                image_embed_step,  # [_ l d], including hard negatives
                video_embed_step,  # [_ nfl d]
                selfattn_mask_step,  # [_ l], including CLS
                crossattn_mask_step,  # [_ 1 l nf*l]
                output_attentions=True,
            ) 
            #retrieve_logits_step = self.heads["vim"](pooled_output).squeeze(-1).view(STEP_SIZE, NUM_SAMPLE)
            
            retrieve_logits_step = F.normalize(pooled_output, dim=-1) @ F.normalize(self.heads["vim"].weight, dim=-1).t() * 10.0
            retrieve_logits_step = retrieve_logits_step.squeeze(-1).view(STEP_SIZE, NUM_SAMPLE)
 
            #retrieve_logits_step = self.heads["vim"](pooled_output)  # (STEP_SIZE*NUM_SAMPLE, 2)
            pairwise_target_step = torch.zeros(
                    (STEP_SIZE, NUM_SAMPLE),
                    dtype=torch.long,
                    device=retrieve_logits_step.device
            )
            pairwise_target_step[:, 0] = 1

            attn_weights = [
                t.view(STEP_SIZE, NUM_SAMPLE, *t.shape[1:])[:,0, ...] for t in attn_weights
            ]  # only retain positive attn_weights, [STEP_SIZE num_heads l s]

            retrieve_logits_lst.append(retrieve_logits_step)
            pairwise_target_lst.append(pairwise_target_step)
            retrieve_attn_weights_lst.append(attn_weights)

        retrieve_logits = torch.cat(retrieve_logits_lst, dim=0)  # [bsz NUM_SAMPLE]
        #retrieve_weights = torch.cat(retrieve_attn_weights_lst, dim=0)

        pairwise_target = torch.cat(pairwise_target_lst, dim=0).view(-1)  # [bsz NUM_SAMPLE] -> [bsz*NUM_SAMPLE]
        #print(retrieve_logits[:2,:])
        #print(pairwise_target)
        sample_list.targets = pairwise_target

        model_output = {
                "scores": retrieve_logits,
                "diag": False,
        }
        loss = {}
        loss["vim_loss"] = self.loss_funcs["vim"](sample_list, model_output)
        model_output["losses"] = loss

        return model_output

    def _forward_vim_selfattn(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.get_hard_pairs_video2image(sample_list)
        #pooled_output = self.transformer.get_joint_embedding_video2image(
        #    sample_list.image_patch,  # [bsz l d], including hard negatives
        #    sample_list.video_patch,  # [bsz nf*l d]
        #    sample_list.attention_mask,  # [bsz nf*l+1], including CLS
        #)
        pooled_output, attentions = self.get_joint_embedding_cross_video2image(
            sample_list.image_patch_hard,  # [bsz l d], including hard negatives
            sample_list.video_patch,  # [bsz nf*l d]
            sample_list.self_attention_mask,  # [bsz l+1], including CLS
            sample_list.cross_attention_mask,  # [bsz 1 l+1 nf*l]
            output_attentions=True,
        )
        sample_list.attention_weights = attentions
       
        logits = self.heads["vim"](pooled_output)
        reshaped_logits = logits.contiguous().view(-1, 2)
        output_dict = {"scores": reshaped_logits}

        loss = {}
        loss["vim_loss"] = self.loss_funcs["vim"](sample_list, output_dict)

        output_dict["losses"] = loss
        return output_dict

    def _forward_rec(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        _, attentions = self.get_joint_embedding_cross_video2image(
            sample_list.image_patch,  # [bsz l d]
            sample_list.video_patch,  # [bsz nf*l d]
            sample_list.self_attention_mask,  # [bsz l+1], including CLS
            sample_list.cross_attention_mask,  # [bsz 1 l+1 nf*l]
            output_attentions=True,
        )

        x = sample_list.video_patch  # [bsz nf*l d], w/o CLS
        y = sample_list.image_patch  # [bs l d], w/o CLS  
        src_len = x.size(1)
        n_frame = sample_list.video_temporal_mask.size(1)
        n_video_patch = src_len // n_frame
        video_mask = sample_list.video_temporal_mask[:,:,None,None].expand(-1,-1,n_video_patch,-1)  # [bsz nf] -> [bsz, nf, 1, 1] -> [bsz, nf, l, 1]
        video_mask = torch.flatten(video_mask, 1, 2)
        x = x * video_mask

        # attentions is a tuple, attns from all transformer layer 
        attentions = [t for index, t in enumerate(attentions) if index%2==1]  # access cross attention while ignore self attention
        attentions = torch.cat(attentions, dim=1)  # [bsz num_heads*num_cross_layers q_len kv_len], cat along num_heads
        #attentions = attentions[:,:,1:,:]  # ignore CLS
        
        #tmp = attentions[:,0,24,:]
        #print("---------------------------------------------> before conv")
        #print(tmp.max(dim=-1)[0], tmp.min(dim=-1)[0], tmp.mean(dim=-1))

        attentions = self.heads["rec"](attentions)  # [bsz 1 q_len kv_len]
        attentions = attentions.squeeze(1)  # [bsz q_len kv_len]
        attentions = nn.functional.softmax(attentions, dim=-1)

        tmp = attentions[:2,24,:]
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> after conv")
        print(tmp.max(dim=-1)[0], tmp.min(dim=-1)[0])
        
        #logit_scale = 4.0
        #fake_y = torch.bmm(attentions, x).mean(dim=1) # [bsz q_len d]
        #fake_y = nn.functional.normalize(fake_y, dim=-1) * logit_scale
        #y = y.mean(dim=1).detach()  # [bsz d]
        #y = nn.functional.normalize(y, dim=-1)  * logit_scale
        
        fake_y = torch.bmm(attentions, x.detach())  # [bsz q_len d]
        y = y.detach()  # [bsz q_len d]

        output_dict = {
                "scores": fake_y,
                "targets": y
        }

        loss = {}
        loss["rec_loss"] = self.loss_funcs["rec"](output_dict)
        output_dict["losses"] = loss
        return output_dict

    def _forward_mlm(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sequence_output, _, _ = self.get_joint_embedding(
            sample_list["input_ids_masked"],
            sample_list["segment_ids"],
            sample_list["image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        num_visual_tokens = sample_list["image"].shape[1]
        sequence_output = sequence_output[:, :-num_visual_tokens]
        logits = (
            self.heads["mlm"](sequence_output)
            .contiguous()
            .view(-1, self.fusion.config.vocab_size)
        )
        labels = sample_list["lm_label_ids"].contiguous().view(-1)
        sample_list["targets"] = labels

        output_dict = {"scores": logits}

        loss = {}
        loss["mlm_loss"] = self.loss_funcs["mlm"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_mpfr(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        hidden, _, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["masked_image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        _, num_visual_tokens, visual_dim = sample_list["image"].shape

        mask = sample_list["image_masks"] == 1
        mask = mask.unsqueeze(-1)

        hidden = hidden[:, -num_visual_tokens:]
        hidden_masked = (
            hidden[mask.expand_as(hidden)].contiguous().view(-1, hidden.size(-1))
        )
        predict_feat = self.heads["mpfr"](hidden_masked)

        target = sample_list["image"]
        target_masked = target[mask.expand_as(target)].contiguous().view(-1, visual_dim)

        sample_list["targets"] = target_masked.detach()

        output_dict = {"scores": predict_feat}

        loss = {}
        loss["mpfr_loss"] = self.loss_funcs["mpfr"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_mpfc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        hidden, _, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["masked_image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        _, num_visual_tokens, visual_dim = sample_list["image"].shape

        mask = sample_list["image_masks"] == 1

        hidden = hidden[:, -num_visual_tokens:]
        hidden_masked = (
            hidden[mask.unsqueeze(-1).expand_as(hidden)]
            .contiguous()
            .view(-1, hidden.size(-1))
        )
        logits = self.heads["mpfc"](hidden_masked)

        target = self.get_patch_labels(sample_list["original_image"])
        target_masked = target[mask].contiguous().view(-1)

        sample_list["targets"] = target_masked

        output_dict = {"scores": logits}

        loss = {}
        loss["mpfc_loss"] = self.loss_funcs["mpfc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_2wpa(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.get_hard_pairs(sample_list)
        visual_embeddings, _, _ = self.get_image_embedding(
            sample_list["image"], sample_list["visual_embeddings_type"]
        )
        image_pad = torch.zeros_like(sample_list["visual_embeddings_type"]).bool()

        text_embeddings, _, _ = self.get_text_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["input_mask"],
        )
        text_pad = ~sample_list["input_mask"].bool()

        ot_dist = optimal_transport_dist(
            text_embeddings, visual_embeddings, text_pad, image_pad
        ).to(text_embeddings.device)

        itm_labels = sample_list["targets"]
        ot_pos = ot_dist.masked_select(itm_labels == 1)
        ot_neg = ot_dist.masked_select(itm_labels == 0)
        ot_loss = (ot_pos.sum() - ot_neg.sum()) / (ot_pos.size(0) + ot_neg.size(0))

        loss = {}
        loss["2wpa_loss"] = ot_loss
        output_dict = {}
        output_dict["losses"] = loss

        return output_dict

    def _forward_mvc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.get_image_embedding(
            sample_list["image_0"], sample_list["visual_embeddings_type"]
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.contrastive_norm(visual_embeddings)

        text_embeddings, _, _ = self.get_text_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["input_mask"],
        )
        masks = sample_list["input_mask"]
        text_embeddings = text_embeddings * masks.unsqueeze(2)
        text_embeddings = torch.sum(text_embeddings, dim=1) / (
            torch.sum(masks, dim=1, keepdim=True)
        )
        text_embeddings = self.contrastive_norm(text_embeddings)

        comp_embeddings, _, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["image_1"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        num_visual_tokens = sample_list["image_1"].shape[1]
        comp_embeddings = comp_embeddings[:, -num_visual_tokens:].mean(dim=1)
        comp_embeddings = self.contrastive_norm(comp_embeddings)

        output_dict = {
            "scores": torch.cat(
                (visual_embeddings, text_embeddings, comp_embeddings), dim=0
            ),
        }
        sample_list["targets"] = sample_list["ann_idx"].squeeze().repeat(3)

        loss = {}
        loss["mvc_loss"] = self.loss_funcs["mvc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_pac(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.get_image_embedding(
            sample_list["image"], sample_list["visual_embeddings_type"]
        )
        visual_embeddings = visual_embeddings.mean(dim=1)

        text_embeddings, _, _ = self.get_text_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["input_mask"],
        )
        masks = sample_list["input_mask"]
        text_embeddings = text_embeddings * masks.unsqueeze(2)
        text_embeddings = torch.sum(text_embeddings, dim=1) / (
            torch.sum(masks, dim=1, keepdim=True)
        )

        visual_logits = self.heads["pac"](visual_embeddings)
        text_logits = self.heads["pac"](text_embeddings)
        sample_list.targets = sample_list.attribute_labels

        loss = {}
        output_dict = {"scores": visual_logits}
        loss["pac_loss"] = self.loss_funcs["pac"](sample_list, output_dict)
        output_dict = {"scores": text_logits}
        loss["pac_loss"] += self.loss_funcs["pac"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_icc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.get_image_embedding(
            sample_list["image"], sample_list["visual_embeddings_type"]
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.contrastive_norm(visual_embeddings)

        num_visual_tokens = sample_list["dv_image"].shape[1]
        num_text_tokens = sample_list["input_ids"].shape[1]
        dropout_ratio = 0.25
        text_dropout_index = torch.randint(
            high=num_text_tokens, size=(int(dropout_ratio * num_text_tokens),)
        )
        pacth_dropout_index = (
            torch.randint(
                high=num_visual_tokens, size=(int(dropout_ratio * num_visual_tokens),)
            )
            + num_text_tokens
        )
        sample_list["attention_mask"][:, text_dropout_index] = 0
        sample_list["attention_mask"][:, pacth_dropout_index] = 0
        comp_embeddings, _, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["dv_image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        comp_embeddings[:, pacth_dropout_index] = 0
        comp_embeddings = comp_embeddings[:, -num_visual_tokens:].sum(dim=1) / (
            num_visual_tokens - int(dropout_ratio * num_visual_tokens)
        )
        comp_embeddings = self.contrastive_norm(comp_embeddings)

        output_dict = {
            "scores": visual_embeddings,
            "targets": comp_embeddings,
        }

        loss = {}
        loss["icc_loss"] = self.loss_funcs["icc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict
    
    def __forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Flatten and prepare mask
        sample_list = self.preprocessing_input(sample_list)
        
        # Forwarding tasks
        if sample_list["task"] == "vim":
            output_dict = self._forward_vim(sample_list)
        elif sample_list["task"] == "rec":
            output_dict = self._forward_rec(sample_list)
        elif sample_list["task"] == "mlm":
            output_dict = self._forward_mlm(sample_list)
        elif sample_list["task"] == "mpfr":
            output_dict = self._forward_mpfr(sample_list)
        elif sample_list["task"] == "mpfc":
            output_dict = self._forward_mpfc(sample_list)
        elif sample_list["task"] == "2wpa":
            output_dict = self._forward_2wpa(sample_list)
        elif sample_list["task"] == "mvc":
            output_dict = self._forward_mvc(sample_list)
        elif sample_list["task"] == "pac":
            output_dict = self._forward_pac(sample_list)
        elif sample_list["task"] == "icc":
            output_dict = self._forward_icc(sample_list)
        else:
            output_dict = self._forward_vic(sample_list)

        return output_dict

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Flatten and prepare mask
        sample_list = self.preprocessing_input(sample_list)
        
        # Forwarding tasks
        model_output = self._forward_vic(sample_list)
        if not self.training:
            return model_output
        
        if hasattr(self.heads, "vim"):
            vim_loss = self._forward_vim(sample_list)["losses"]
            model_output["losses"].update(vim_loss)

        if hasattr(self.heads, "rec"):
            rec_loss = self._forward_rec(sample_list)["losses"]
            model_output["losses"].update(rec_loss)

        return model_output
