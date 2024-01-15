

# Last Change:  2023-12-17 03:42:57

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import random
from pathlib import Path

import torch
from torch import nn

from module.util_module import PreTrainedModel, AllGather, CrossEn, ReconsWeight
from module.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from module.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)
allgather = AllGather.apply
 
try:
    local_rank = torch.distributed.get_rank()
except:
    local_rank = 0

def show_log(info):
    logger.warning(info)

def _update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log("Set {}.{}: {}.".format(target_name, target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def update_attr(target_name, target_config, target_attr_name, default_value=None):
    if default_value is not None:
        setattr(target_config, target_attr_name, default_value)
        show_log("Set {}.{}: {}.".format(target_name, target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class CLIP4Clip(PreTrainedModel, nn.Module):
    def __init__(self, 
            cross_config, 
            clip_state_dict, 
            max_words, 
            max_frames, 
            one_stage,
            linear_patch, 
            sim_header, 
            cross_num_hidden_layers, 
            recons_feat, 
            embedding_sim, 
            add_text
            ):
        super(CLIP4Clip, self).__init__(cross_config)
        self.ignore_video_index = -1

        assert max_words + max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log("Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.one_stage = False
        if self._stage_one and one_stage:
            self.one_stage = True
            show_log("Test retrieval by one stage.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log("\t embed_dim: {}".format(embed_dim))
        show_log("\t image_resolution: {}".format(image_resolution))
        show_log("\t vision_layers: {}".format(vision_layers))
        show_log("\t vision_width: {}".format(vision_width))
        show_log("\t vision_patch_size: {}".format(vision_patch_size))
        show_log("\t context_length: {}".format(context_length))
        show_log("\t vocab_size: {}".format(vocab_size))
        show_log("\t transformer_width: {}".format(transformer_width))
        show_log("\t transformer_heads: {}".format(transformer_heads))
        show_log("\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = linear_patch
        show_log("\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log("\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]
        
        # Convert to fp16
        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = sim_header
        show_log("\t sim_header: {}".format(self.sim_header))

        if self.sim_header == "cross_attention": assert self.one_stage is False

        cross_config.max_position_embeddings = context_length  # TODO BUG
        #cross_config.max_position_embeddings = (max_frames+1)*((image_resolution//vision_patch_size)**2+1) # MAX cross attention token
        if self.one_stage is False:
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", default_value=cross_num_hidden_layers)
            self.cross = CrossModel(cross_config)
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seq_lstm" or self.sim_header == "seq_transformer":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seq_transformer":
            self.transformerClip = TransformerClip(width=transformer_width, layers=cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seq_lstm":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn()
        self.logit_scale = 100.
        self.recons_feat = recons_feat
        self.embedding_sim = embedding_sim
        if recons_feat:
            # Feature Reconstruction 
            self.recons_weights = ReconsWeight(in_dim=8)
            self.loss_recons = nn.MSELoss()
            
        self.add_text = add_text
        if self.add_text:
            emb_dim = 512
            self.fc_asr = nn.Linear(emb_dim, emb_dim)
            self.fc_title = nn.Linear(emb_dim, emb_dim)

        # Random initialize
        self.apply(self.init_weights)

    @classmethod
    def from_pretrained(cls, 
            cross_model_name, 
            max_words, 
            max_frames, 
            one_stage,
            linear_patch,
            sim_header, 
            pretrained_clip_name,
            cross_num_hidden_layers,
            state_dict=None,
            cache_dir=None,
            type_vocab_size=2,
            training=True,
            recons_feat=False,
            embedding_sim=True,
            add_text=False):

        if state_dict is None:
            state_dict = {}

        pretrained_clip_name = "ViT-B/32" if pretrained_clip_name is None else pretrained_clip_name
        
        # Load ViT parameters
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        logging.info(f"Successully load state_dict: {pretrained_clip_name}")
        for key, val in clip_state_dict.items():
            if key in ["context_length", "input_resolution", "vocab_size"] and (not training):
                continue
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()
        
        # Cross model, input is video and image, outputs similarity. Useless when one_stage is True
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None)

        # Init CLIP4Clip, random initialing parameters
        model = cls(cross_config, clip_state_dict, max_words, max_frames, one_stage, linear_patch, sim_header, cross_num_hidden_layers, recons_feat, embedding_sim, add_text)

        # >>>>>>> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'cross_attention':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seq_lstm" or model.sim_header == "seq_transformer":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seq_transformer" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        # <<<<<<< End of initialization trick
        
        # Loading pretrained parameters
        if state_dict is not None:
            model = cls.load_preweight(model, state_dict, local_rank)
        
        # Print model
        if training:
            with open("./model.txt", "w") as f:
                print(model, file=f)
        return model

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        batch_size = input_ids.size(0)  # batch_size = bs
        sequence_hidden = self.clip.encode_text(input_ids, return_hidden=False).float()  # (bs d)
        sequence_hidden = sequence_hidden.view(batch_size, -1, sequence_hidden.size(-1)) # (bs 1 d)

        return sequence_hidden

    def get_visual_output(self, video, batch_size, video_frame=-1, return_hidden=True, attn_mask=None):
        visual_cls, visual_hidden = self.clip.encode_image(video, video_frame=video_frame, return_hidden=return_hidden, attn_mask=attn_mask)  # (bs*max_frame 512)
        visual_cls, visual_hidden = visual_cls.float(), visual_hidden.float()
        visual_cls = visual_cls.view(batch_size, -1, visual_cls.size(-1))  
        return visual_cls, visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        try:
            visual_output = visual_output * video_mask_un
        except Exception as e:
            print(visual_output.device, video_mask_un.device, next(self.parameters()).device)
            print(e)
            visual_output = visual_output * video_mask_un.to(visual_output.device)
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, visual_output, video_mask, sim_header="mean_pooling"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        #if sim_header == "mean_pooling":
        #    # Default: Parameter-free type
        #    pass
        #elif sim_header == "seq_lstm":
        #    # Sequential type: LSTM
        #    visual_output_original = visual_output
        #    visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
        #                                         batch_first=True, enforce_sorted=False)
        #    visual_output, _ = self.lstm_visual(visual_output)
        #    if self.training: self.lstm_visual.flatten_parameters()
        #    visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
        #    visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
        #    visual_output = visual_output + visual_output_original
        #elif sim_header == "seq_transformer":
        #    # Sequential type: Transformer Encoder
        #    visual_output_original = visual_output
        #    seq_length = visual_output.size(1)
        #    position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        #    position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        #    frame_position_embeddings = self.frame_position_embeddings(position_ids)
        #    visual_output = visual_output + frame_position_embeddings

        #    extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        #    extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        #    visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        #    visual_output = self.transformerClip(visual_output, extended_video_mask)
        #    visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        #    visual_output = visual_output + visual_output_original

        if self.training:
            visual_output = allgather(visual_output)
            video_mask = allgather(video_mask)
            sequence_output = allgather(sequence_output)
            torch.distributed.barrier()

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        #logit_scale = self.clip.logit_scale.exp()
        logit_scale = self.logit_scale if self.training else 1.
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def _cross_similarity(self, sequence_output, visual_output, video_mask, batch_simmat=None):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        if self.training:
            sequence_output = allgather(sequence_output)
            visual_output = allgather(visual_output)
            video_mask = allgather(video_mask)
            torch.distributed.barrier()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()
        num_sample = 4 if self.training else b_visual # {b_visual, num_sample}, number of positive + negative 

        if batch_simmat is not None:
            assert batch_simmat.size(0) == b_text and batch_simmat.size(1)==b_visual
            _, indexes = batch_simmat.sort(descending=True)
            indexes = indexes.detach().cpu().numpy().tolist()
            hard_index_lst = list()
            for hard_i, _index in enumerate(indexes):
                hard_index_lst.append([t for t in _index if t!=hard_i][:(num_sample-1)])

        retrieve_logits_list = []
        retrieve_attn_weights_list = []

        #step_size = b_text      # set smaller to reduce memory cost
        step_size = 2 # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(b_text, s_text)\
            .to(device=video_mask.device, dtype=video_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, num_sample, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)  # (bs*num_sample s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, num_sample, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            if not self.training:
                # Opt1: Use all video, num_sample=b_visual
                visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
                visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
                video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
                video_mask_r = video_mask_r.view(-1, s_visual)
            else:
                # Opt2: Random or Hard Negative sample
                _tic = i * step_size 
                _toc = (i+1) * step_size
                _toc = _toc if _toc < sequence_output.size(0) else sequence_output.size(0)
                visual_output_r = list()
                video_mask_r = list()
                for j in range(_tic, _toc):
                    # Random sampling
                    #index_neg = list(range(b_visual))
                    #index_neg.pop(j)
                    #index_neg = random.sample(index_neg, num_sample-1)

                    # Hard sampling
                    index_neg = hard_index_lst[j]

                    pos = visual_output[j:j+1]  # (1 S d)
                    neg = visual_output[index_neg]  # (num_sample-1 S d)
                    pair = torch.cat([pos, neg], dim=0)  # (num_sample S d)
                    visual_output_r.append(pair)

                    pos_mask = video_mask[j:j+1]
                    neg_mask = video_mask[index_neg]  # (num_sample-1 S)
                    pair_mask = torch.cat([pos_mask, neg_mask], dim=0)
                    video_mask_r.append(pair_mask)

                visual_output_r = torch.stack(visual_output_r, dim=0)  # (bs num_sample S d)
                visual_output_r = visual_output_r.view(-1, s_visual, h_visual)  # (bs*num_sample S d), view=concat all rows
                video_mask_r = torch.stack(video_mask_r, dim=0)
                video_mask_r = video_mask_r.view(-1, s_visual)


            #cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            # pooled_output (step_truth*num_sample d)
            # attn_weights (step_truth*num_sample num_heads L S)
            pooled_output, attn_weights = self.cross(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, num_sample)
            attn_weights = attn_weights.view(step_truth, num_sample, *attn_weights.shape[1:])[:,0, ...]  # only retain positive attn_weights

            retrieve_logits_list.append(retrieve_logits_row)
            retrieve_attn_weights_list.append(attn_weights)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        retrieve_weights = torch.cat(retrieve_attn_weights_list, dim=0)
        return retrieve_logits, retrieve_weights

    def get_similarity_logits(self, 
            sequence_output, sequence_hidden, 
            visual_output, visual_hidden, video_mask, patch_mask, 
            asr_emb=None, item_emb=None,
            shaped=False, one_stage=False):
        '''
        sequence_output (bs 1 d)
        sequence_hidden (bs 1+num_patch d)
        visual_output (bs num_frame d)
        visual_hidden to (bs num_frame 1+num_patch d)
        video_mask (bs num_frame)
        patch_mask (bs num_frame 1+num_patch)
        asr_emb (bs d)
        item_emb (bs d)
        '''
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        output = dict()
        # contrastive learning logits (bs bs)
        cl_logits = self._loose_similarity(sequence_output, visual_output, video_mask, sim_header=self.sim_header)
        output.update({"contrastive_logit": cl_logits})

        if not one_stage: 
            assert self.sim_header in ["cross_attention"]
            _tic = time.time()
            visual_hidden = visual_hidden.view(visual_hidden.size(0), -1, visual_hidden.size(-1))  # (bs num_frame 1+num_patch d) to (bs num_frame*1+num_patch d)
            patch_mask = patch_mask.view(patch_mask.size(0), -1)  # (bs max_frame 1+num_patch) to (bs max_frame*1+num_patch)
            # Concat text modality
            if self.add_text:
                item_emb = self.fc_title(item_emb.to(sequence_hidden.dtype)).unsqueeze(1)  # (bs d) to (bs 1 d)
                sequence_hidden = torch.cat([sequence_hidden, item_emb], dim=1)  # bs (1+num_patch)+1 d
                asr_emb = self.fc_asr(asr_emb.to(visual_hidden.dtype)).unsqueeze(1)  # (bs d) to (bs 1 d)
                visual_hidden = torch.cat([visual_hidden, asr_emb], dim=1)  # bs max_frame*(1+num_patch)+1 d
                text_mask = torch.ones([patch_mask.size(0), 1], dtype=patch_mask.dtype, device=patch_mask.device)
                patch_mask = torch.cat([patch_mask, text_mask], dim=1)  # (bs max_frame*1+num_patch) to bs max_frame*(1+num_patch)+1
            mt_logits, attn_weights = self._cross_similarity(sequence_hidden, visual_hidden, patch_mask, batch_simmat=cl_logits)
            logit_scale = self.logit_scale if self.training else 1. 
            mt_logits = logit_scale * mt_logits
            _dur = time.time() - _tic
            #print(f"Cross similarity takes time: {_dur:.3f} sec")
            
            if self.add_text:
                attn_weights = attn_weights[:,:,:-1,:-1]  # (bs num_head N+1 M+1) to (bs num_head N M)
            output.update({"pairwise_logit": mt_logits, "attn_weights": attn_weights})

        return output
    
    def forward(self, video, video_mask, image, patch_mask=None, asr_emb=None, item_emb=None):
        """
        video: (bs max_frame 3 H W)
        video_mask: (bs max_frame)
        image: (bs 3 H W)
        patch_mask: (bs max_frame 1+num_patch)
        asr_emb: (bs d)
        item_emb: (bs d)
        """
        #input_ids = input_ids.view(-1, input_ids.shape[-1])  # (bs 1 max_words) to (bs max_words)
        #token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])  # same size as input_ids, all zeros
        #attention_mask = attention_mask.view(-1, attention_mask.shape[-1])  # same size as input_ids, 1 and 0
        batch_size = video.size(0)

        video = torch.as_tensor(video).float()  # (bs max_frame 3 H W)
        bs, ts, channel, h, w = video.shape
        video = video.view(bs*ts, channel, h, w)

        # sequence_output: (bs 1 d)
        # sequence_hidden: (bs 1+num_patch d)
        sequence_output, sequence_hidden = self.get_visual_output(image, batch_size, attn_mask=None)

        if patch_mask is not None: 
            video_attn_mask = patch_mask.view(-1, patch_mask.size(-1)).contiguous()
            video_attn_mask = video_attn_mask==0
            #video_attn_mask = (1 - video_attn_mask).to(torch.uint8)
        else:
            video_attn_mask = patch_mask

        # visual_output: (bs num_frame d)
        # visual_hidden: (bs*num_frame 1+num_patch d) to (bs num_frame 1+num_patch d)
        visual_output, visual_hidden = self.get_visual_output(video, batch_size, attn_mask=video_attn_mask)
        visual_hidden = visual_hidden.view(bs, -1, *visual_hidden.shape[-2:])
        if patch_mask is None:
            patch_mask = video_mask.unsqueeze(-1).repeat(1, 1, sequence_hidden.size(1))  # (bs num_frame 1+num_patch)

        if not self.training:
            return None

        loss_dict = dict()
        output = self.get_similarity_logits(
                sequence_output, sequence_hidden, 
                visual_output, visual_hidden, video_mask, patch_mask,
                asr_emb, item_emb,
                shaped=True, one_stage=self.one_stage
                )

        # Contrastive loss
        sim_matrix = output["contrastive_logit"]  # (bs bs)
        sim_loss1 = self.loss_fct(sim_matrix, diag=True)
        sim_loss2 = self.loss_fct(sim_matrix.T, diag=True)
        sim_loss = (sim_loss1 + sim_loss2) / 2
        loss_dict.update({"contrastive_loss": sim_loss})

        if not self.one_stage:
            # Pairwise loss
            sim_matrix = output["pairwise_logit"]  # (bs 2)
            pairwise_loss = self.loss_fct(sim_matrix, diag=False)
            loss_dict.update({"pairwise_loss": pairwise_loss * 1.0 })
            
            # Reconstruction loss
            if self.recons_feat:
                sequence_hidden = allgather(sequence_hidden)
                visual_hidden = allgather(visual_hidden)
                video_mask = allgather(video_mask)
                if patch_mask is not None:
                    patch_mask = allgather(patch_mask)
                torch.distributed.barrier()
                if patch_mask is not None:
                    visual_hidden = visual_hidden * patch_mask.unsqueeze(-1)  # ((bs num_frame 1+num_patch d))
                visual_hidden = visual_hidden[:, :, 1:, :].contiguous()  # ignore CLS token
                y = visual_hidden.view(visual_hidden.size(0), -1, visual_hidden.size(-1))  # (bs num_frame 1+num_patch d) to (bs num_frame*1+num_patch d)
                
                num_frame = visual_hidden.size(1)
                attn_w = output["attn_weights"]  # (bs num_heads N M)
                attn_w = attn_w.view(attn_w.size(0), attn_w.size(1), attn_w.size(2), num_frame, -1) # (bs num_heads N num_frame N)
                attn_w = attn_w[:, :, 1:, :, 1:].contiguous()
                attn_w = attn_w.view(attn_w.size(0), attn_w.size(1), attn_w.size(2), -1)  # ignore CLS token
                attn_w = self.recons_weights(attn_w).squeeze(1)  # (bs 1 N M) to (bs N M)

                lambda_x = torch.bmm(attn_w, y.detach())
                #lambda_x = torch.bmm(attn_w, y)
                sequence_hidden = sequence_hidden[:, 1:, :].detach()  # ignore CLS token

                diff_loss = self.loss_recons(lambda_x, sequence_hidden)
                loss_dict.update({"diff_loss": diff_loss * 0.05})
                #l1_loss = torch.abs(attn_w).sum(dim=2).mean()
                #loss_dict.update({"diff_loss": diff_loss * 0.05, "l1_loss": l1_loss * 0.02})
        return loss_dict
        
if __name__ == "__main__":
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE', Path.home() / '.pytorch_pretrained_bert'))
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

    pretrained_clip_name = "ViT-B/32"
    model = CLIP4Clip.from_pretrained(cross_model_name="cross-base", 
            max_words=32, max_frames=12, one_stage=True, linear_patch="2d", sim_header="mean_pooling", 
            pretrained_clip_name=pretrained_clip_name, cross_num_hidden_layers=4,
            state_dict=None, cache_dir=cache_dir, type_vocab_size=2)
    print(model)

    # model.clip, model.loss_fct
    # model.clip.visual, model.clip.transformer (encode text)
    #with open("./model.txt", "w") as f:
    #    print(model, file=f)
