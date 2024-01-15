

# Last Change:  2023-11-05 03:05:07

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import math
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json
import random
import logging
from dataset.rawframe_util import RawVideoExtractor
from module.util_module import genPatchmask, IOU

class LPR4M_TrainDataLoader(Dataset):
    """LPR4M train dataset loader."""
    def __init__(
            self,
            data_root,
            data_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=10,
            image_resolution=224,
            frame_order=0,
            slice_framepos=2,
            unfold_sentences=False,
    ):
        dataroot = os.path.dirname(data_path)
        # Generate grid
        patchsize = (32, 32)
        h, w = patchsize
        grid_h = image_resolution // h
        grid_w = image_resolution // w
        self.grids = []
        for i in range(grid_h):
            for j in range(grid_w):
                ltrb = [j*w, i*h, (j+1)*w, (i+1)*h]    
                self.grids.append(ltrb)
                
        # Loading asr embedding file
        #with open(os.path.join(dataroot, "training_seg2emb_65w.json"), "r") as f:
        #    self.seg2emb = json.load(f)
        #print(f"Successfully load seg2emb: {len(self.seg2emb)}")
        
        # Loading asr embedding file
        #with open(os.path.join(dataroot, "training_item2emb_155100.json"), "r") as f:
        #    self.item2emb = json.load(f)
        #print(f"Successfully load item2emb: {len(self.item2emb)}")
        
        # Training file        
        self.training = True if "training" in data_path else False
        self.data = open(data_path).readlines()
        
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]

        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.sample_len = 0
        
        self.sample_info = {}
        for _line in tqdm(self.data, desc="formating data ..."):
            line = _line.strip().split("\t")
            clip_id = line[0]
            frame_path_lst = eval(line[1])
            frame_path_lst = [os.path.join(data_root, t) for t in frame_path_lst]
            image_id = line[3]
            image_path = line[4]
            image_path = os.path.join(data_root, image_path)
            #spu_id = line[12]
            
            # Generating patch mask
            #hw_lst = eval(line[2])
            #hw_lst = [[int(_t) for _t in t.split("\x01")] for t in hw_lst]
            #frame_box_lst = eval(line[3])
            #frame_box_lst = [[t[0]/_hw[1], t[1]/_hw[0], t[2]/_hw[1], t[3]/_hw[0]] for t, _hw in zip(frame_box_lst, hw_lst)]
            #frame_box_lst = [t if sum(t)>0. else [0.,0.,1.,1.] for t in frame_box_lst]
            #frame_box_lst = [[t[0]*image_resolution, t[1]*image_resolution, t[2]*image_resolution, t[3]*image_resolution] for t in frame_box_lst]
            #patch_mask = [genPatchmask([t], self.grids, iou_thresh=0.02) for t in frame_box_lst]
            #patch_mask = np.array(patch_mask, dtype=np.long)
            patch_mask = None
            
            self.sample_info[len(self.sample_info)] = (clip_id, frame_path_lst, image_path, patch_mask, clip_id, image_id)

        self.sample_len = len(self.sample_info)
        
        self.rawVideoExtractor = RawVideoExtractor(centercrop=True, size=(image_resolution, image_resolution))
        self.rawImageExtractor = RawVideoExtractor(centercrop=True, size=(image_resolution, image_resolution))
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.mask_percent = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.mask_or_not = np.arange(0., 1., 0.1).tolist()
        
        
    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words
    
    def _get_image(self, image_path):
        raw_image_data = self.rawImageExtractor.get_video_data([image_path])
        return raw_image_data["video"]
    
    def _get_rawvideo(self, framepath_lst, choice_video_ids):
        num_video = len(choice_video_ids)
        video_mask = np.zeros((num_video, self.max_frames), dtype=np.long)
        max_video_length = [0] * num_video

        video = np.zeros((num_video, self.max_frames, 1, 3,
                          self.rawVideoExtractor.size_h, self.rawVideoExtractor.size_w), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            raw_video_data = self.rawVideoExtractor.get_video_data(framepath_lst[i])
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # num_frame x 1 x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice
                
                # num_frame x 1 x 3 x H x W
                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)
                if self.training and random.sample(self.mask_or_not, 1)[0]>=0.5:
                    _percent = random.sample(self.mask_percent, 1)[0]
                    num_keep = video_slice.size(0) - math.floor(video_slice.size(0)*_percent)
                    assert num_keep > 0
                    video_slice = video_slice[:num_keep, ...]
                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        while True:
            try:
                assert idx >= 0 and idx < len(self), "Provided idx ({}) is outside of dataset range".format(_index)
                clip_id, framepath_lst, image_path, patch_mask, clip_id, image_id = self.sample_info[idx]  # patch_maks (num_frame num_patch)
                
                # Image and video
                #pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
                # image: (1 3 h w)
                image = self._get_image(image_path)
                image = image.squeeze(0)
                # video: (1 max_frame, 1, 3, h, w)
                # video_mask: (1 max_frame)
                video, video_mask = self._get_rawvideo([framepath_lst], [clip_id])  # video_mask (1 num_frame)
                video = video.squeeze(2).squeeze(0)  # (max_frame 3 h w)
                video_mask = video_mask.squeeze(0)
                #patch_mask = patch_mask*video_mask.squeeze(0)[:, np.newaxis]
                
                # ASR embedding
                #asr_emb = self.seg2emb[clip_id]
                #asr_emb = np.array(asr_emb)
                
                # Title embedding
                #item_emb = self.item2emb[image_id]
                #item_emb = np.array(item_emb)
            
                break
            except Exception as e:
                logging.info(f"!!! Meeting bad data: line: {idx}, error: {e}")
                idx = idx - 1
        #return video, video_mask, patch_mask, asr_emb, image, item_emb
        return video, video_mask, image

class LPR4M_DataLoader(LPR4M_TrainDataLoader):
    """LPR4M dataset loader."""
    def __init__(
            self,
            data_root,
            data_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=10,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0):
        super(LPR4M_DataLoader, self).__init__(
            data_root,
            data_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=10,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0)
