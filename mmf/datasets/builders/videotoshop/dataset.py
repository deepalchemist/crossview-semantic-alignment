

# Last Change:  2024-05-05 05:56:07

import copy
import json
import random
import logging

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.mmf_dataset import MMFDataset
from .database import VideoToShopDatabase
logger = logging.getLogger(__name__)

class VideoToShopDataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        '''
        Input:
            dataset_type: train | val | test
            index: supporting multi-dataset train|val|test, index of datasets,
                   init dataset one-by-one then merge datasets.
        '''
        super().__init__(
            "videotoshop",
            config,
            dataset_type,
            index,
            VideoToShopDatabase,
            *args,
            **kwargs,
        )
        self._use_video = self.config.get("use_video", True)
        self._double_view = config.get("double_view", False)
        self._attribute_label = config.get("attribute_label", False)
        self._category_label = config.get("category_label", False)
        self._subcategory_label = config.get("subcategory_label", False)

    def init_processors(self):
        super().init_processors()
        if self._use_images:
            # Assign transforms to the image_db
            # FROM mmf/configs/dataset/dataset_name/default.yaml
            if self._dataset_type == "train":
                self.image_db.transform = self.train_image_processor
            else:
                self.image_db.transform = self.eval_image_processor

    def _get_valid_text_attribute(self, sample_info):
        if "captions" in sample_info:
            return "captions"

        if "sentences" in sample_info:
            return "sentences"

        raise AttributeError("No valid text attribution was found")

    def getitem(self, idx):
        # get output of _load_annotation_db in database.py
        sample_info = self.annotation_db[idx]  
        text_attr = self._get_valid_text_attribute(sample_info)  # get caption/sentences

        current_sample = Sample()
        sentence = sample_info[text_attr]
        current_sample.text = sentence

        if hasattr(self, "masked_token_processor") and self._dataset_type == "train":
            processed_sentence = self.masked_token_processor({"text": sentence})
            '''
            tokens: [CLS] sentence [SEP]
            tokens_masked: [CLS] sentence with [MASK] [SEP]
            input_ids: mapping token word to token ID
            input_ids_masked: mapping tokens_masked word to ID
            lm_label_ids: label for mlm, present masked token id, other is -1
            input_mask: 1/0 mask, [0,len(tokens)] is 1, other is 0, len=max_sentence_len
            segment_ids: token type ID, default is 0
            '''
            current_sample.update(processed_sentence)
        else:
            processed_sentence = self.text_processor({"text": sentence})
            '''
            'input_ids', 'input_mask', 'segment_ids', 'lm_label_ids', 'tokens', 'text'
            '''
            current_sample.update(processed_sentence)
        
        if self._use_video:
            current_sample.video_id = sample_info["video_id"]
            video_path = sample_info["video_path"]
            video = self.image_db.from_path(video_path)["images"]  # List, normalized float tensor, transformed by train_image_processor in dataset yaml
            video = torch.stack(video, dim=0)  # [nframe 3 h w]
            # processing temporal masks
            assert hasattr(self, "masked_video_processor")
            # NOTE: 1 is to keep, 0 is to mask
            current_sample.video, video_temporal_mask = self.masked_video_processor(video)  # video: [MAX 3 img_h img_w], mask: [MAX,]
            current_sample.video_temporal_mask = video_temporal_mask

            if hasattr(self, "masked_image_processor"):  # FIXME: do nothing for image 
                use_augments = True if self._dataset_type == "train" else False
                video_spatial_mask = [self.masked_image_processor(_frame, use_augments=use_augments) for _frame in current_sample.video]
                video_spatial_mask = torch.stack(video_spatial_mask)  # [MAX h w]
                current_sample.video_masks = current_sample.video_temporal_mask[:,None,None]*video_spatial_mask
            assert current_sample.video.size(0) == 10, f"Video ID: {sample_info['video_id']}, Number frames is {current_sample.video.size(0)}"

        if self._use_images:
            image_path = sample_info["image_path"]
            current_sample.image_id = sample_info["image_id"]
            current_sample.image = self.image_db.from_path(image_path)["images"][0]
            if hasattr(self, "masked_image_processor"):  # FIXME: disable when test
                use_augments = True if self._dataset_type == "train" else False
                # NOTE: 1 is to keep, 0 is to mask
                current_sample.image_masks = self.masked_image_processor(current_sample.image, use_augments=use_augments)  # [h w] FIXME: original image have not been masked
 
        elif self._use_features:
            feature_path = ".".join(image_path.split(".")[:-1]) + ".npy"
            current_sample.image = self.features_db.from_path(feature_path)["image_feature_0"]
        if self._dataset_type == "train":
            if self._attribute_label:
                attribute_labels = torch.zeros(2232)
                if len(sample_info["attributes_id"]) > 0:
                    attribute_labels[sample_info["attributes_id"]] = 1 / len(sample_info["attributes_id"])
                current_sample.attribute_labels = attribute_labels
            if self._category_label:
                current_sample.targets = torch.tensor(sample_info["category_id"], dtype=torch.long)
            elif self._subcategory_label:
                current_sample.targets = torch.tensor(sample_info["subcategory_id"], dtype=torch.long)
            else:
                current_sample.targets = torch.tensor(1, dtype=torch.long)
        elif sample_info["testset_type"] == "query":
            current_sample.live_id = sample_info["live_id"]
        elif sample_info["testset_type"] == "gallery":
            current_sample.live_id = sample_info["live_id"]
        else:
            raise NotImplementedError

        current_sample.ann_idx = torch.tensor(idx, dtype=torch.long)
        return current_sample

    def __getitem__(self, idx):
        try_cnt = -1
        while True:
            try:
                try_cnt += 1
                current_sample = self.getitem(idx)
                break
            except Exception as e:
                logger.info(f"!!! Meeting bad data: line: {idx}, error: {e}, Retry times: {try_cnt}")
                #idx = idx  # try again, i.e., Input/Output Error
                idx = idx + 1
        return current_sample
