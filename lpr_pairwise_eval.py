# -*- coding: utf-8 -*-
# hokkien.ywj@gmail.com @2024-06-04 10:07:18

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
try:
    import pickle5 as pkl
except:
    import pickle as pkl

import torch
from torch.utils.data import DataLoader

from mmf.common.sample import Sample
from mmf.models.rice.rice import RICE
from mmf.common.registry import registry
from mmf.utils.configuration import load_yaml
from mmf.datasets.builders.videotoshop.dataset import VideoToShopDataset

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True, type=str, help='path/to/config/file')
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    return args

def check_image_file(filename):
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])

def most_freq(List):
    return max(set(List), key = List.count)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def calculate_acc(pred, gt, top_k=10, root_dir=None, write_dir=None, output_badcase=False):
    n_query = len(gt)
    hit_top = np.array([1, 3, 5, 10])
    hit_top_cnt = np.array([0, 0, 0, 0])
    print('Cal results...')
    for q_index, (q_name, q_pred) in enumerate(pred.items()):
        #if (q_index + 1) % 1000 == 0:
        #    print('Processing {} queries'.format(q_index))
        if isinstance(q_pred[0], str):
            q_pred = [q_pred]
        tmp_top_hit = np.array([0, 0, 0, 0])
        for k in range(top_k):

            if k > len(q_pred)-1:
                cur_pred = q_pred[-1][0]
            else:
                cur_pred = q_pred[k][0]

            if cur_pred in gt[q_name]:
                if k < hit_top[0]:
                    tmp_top_hit[0:] = 1
                elif k < hit_top[1]:
                    tmp_top_hit[1:] = 1
                elif k < hit_top[2]:
                    tmp_top_hit[2:] = 1
                elif k < hit_top[3]:
                    tmp_top_hit[3:] = 1

        hit_top_cnt += tmp_top_hit

    print('Top{} Hit: {:.2f}%'.format(hit_top[0], (100 * hit_top_cnt[0] / n_query)))
    print('Top{} Hit: {:.2f}%'.format(hit_top[1], float(100 * hit_top_cnt[1] / n_query)))
    print('Top{} Hit: {:.2f}%'.format(hit_top[2], float(100 * hit_top_cnt[2] / n_query)))
    print('Top{} Hit: {:.2f}%'.format(hit_top[3], float(100 * hit_top_cnt[3] / n_query)))
    
    if not output_badcase:
        return

    # Record badcase
    dst_file = open(os.path.join(write_dir, "badcase_emb.txt"), "w")
    bbox_path = os.path.join(write_dir, "frame2bbox.json")
    with open(bbox_path, "r") as f:
        f2bbox = json.load(f)

    for q_name, q_pred in pred.items():
        is_hit = False
        top1_item, top1_score = q_pred[0]
        gt_item_lst = gt[q_name]
        if top1_item in gt_item_lst:  # top1 hit
            is_hit = True

        if is_hit:
            pass
        else:
            liveid, timestamp = q_name.split("_")
            frame_dir = os.path.join(root_dir, liveid, "frame")
            item_dir = os.path.join(root_dir, liveid, "shelf")
            frame_path = os.path.join(frame_dir, f"{timestamp}.jpg")
            try:
                frame_bbox = f2bbox[q_name]
            except Exception as e:
                print(e)
                frame_bbox = []
            top1_path = os.path.join(item_dir, f"{top1_item}_0.jpg")
            gt_path = os.path.join(item_dir, f"{gt_item_lst[0]}_0.jpg")
            dst_file.write(f"{frame_path}\t{frame_bbox}\t{top1_path}\t{top1_score}\t{gt_path}\n")
    dst_file.close()
    return

def predict_topk(config, output_root, cdt_video):
    clip_lst = [t for t in os.listdir(output_root) if t.startswith("q2g_simmat_")]
    
    # Merge subdict
    pred = dict()
    for clip_name in tqdm(clip_lst, desc="merge query video results"):
        try:
            dstpath = os.path.join(output_root, clip_name)
            with open(dstpath, "r") as f:
                q2g = json.load(f)
            for _key, _val in q2g.items():
                #_val = [['-'.join(t[0].split("-")[:-1]), t[1]] for t in _val]
                pred[_key] = _val
        except Exception as e:
            print(e)
            print(f"! Error video: {clip_name}")
    print(f"Total number of clip: {len(pred)}")
    
    data_root = config.dataset_config.videotoshop.data_dir
    gt_path = os.path.join(data_root, "annotations/test_videoid_to_gtimage_20079.json")
    with open(gt_path, "r") as f:
        gt = json.load(f)

    # Conditional 
    if cdt_video is not None:
        gt = {k:v for k,v in gt.items() if k in cdt_video}

    gt = {k:v for k,v in gt.items() if k in pred}
    print(f"Total number of gt: {len(gt)}")
    
    calculate_acc(pred, gt, top_k=10, root_dir=None, write_dir=None, output_badcase=False)
    return

def init_model(config):
    model_name = config.model
    model_config = getattr(config.model_config, model_name)
    model_class = registry.get_model_class(model_name)
    model = model_class(model_config)
    model.build()

    # Load state_dict
    ckpt_path = os.path.join(config.env.save_dir, f"{model_name}_final.pth")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    fmt_state_dict = OrderedDict()
    for _key, _val in state_dict.items():
        if _key.startswith("module."):
            _key = _key.replace("module.", "")
        fmt_state_dict[_key] = _val
    model.load_state_dict(fmt_state_dict, strict=True)
    print(f"Successfully load checkpoint from: {ckpt_path}")
    model.eval()
    return model

def pairwise_forward(config, output_root, device_id):
    GALLERY_BATCH_SIZE = 128
    device = torch.device(f"cuda:{device_id}")

    # Init model
    model = init_model(config)
    model = model.to(device)
 
    # Init dataloader
    query_set = VideoToShopDataset(
            config.dataset_config.videotoshop, 
            dataset_type="test",
            index=0,
            )
    query_set.init_processors()
    query_loader = DataLoader(
        query_set,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        )
    
    # Gallery set
    gallery_set = VideoToShopDataset(
        config.dataset_config.videotoshop, 
        dataset_type="test",
        index=1,
        )
    gallery_set.init_processors()
    gallery_loader= DataLoader(
            gallery_set,
            batch_size=GALLERY_BATCH_SIZE,
            num_workers=8,
            shuffle=False,
            drop_last=False,
    )

    _cnt = -1
    for q_sample in tqdm(query_loader, desc=f"query forward"):
        # video: (batch_size max_frame 3 h w)
        # video_temporal_mask: (batch_size max_frame)
        _cnt += 1
        if _cnt == 2: break
        q_sample = Sample(q_sample)
        assert q_sample.video.size(0) == 1
        q_liveid = q_sample.live_id[0]
        q_videoid = q_sample.video_id[0]

        retrieve_logits_full = list()
        imageid_full = list()

        _cnt_g = -1
        for g_sample in tqdm(gallery_loader, desc="gallery forward"):
            _cnt_g += 1
            #if _cnt_g == 5: break
            g_sample = Sample(g_sample)

            # filter 
            retain_index = [_idx for _idx, t in enumerate(g_sample.live_id) if t==q_liveid]
            if len(retain_index) == 0:
                continue

            imageid_lst = [g_sample.image_id[t] for t in retain_index] 
            g_sample.image = g_sample.image[retain_index]
            
            q_sample.image = g_sample.image
            sample_list = q_sample
            sample_list.task = "vim"

            with torch.no_grad():
                sample_list.image = sample_list.image.to(device)
                sample_list.video = sample_list.video.to(device)
                sample_list.video_temporal_mask = sample_list.video_temporal_mask.to(device)

                # Pairwise forward
                sample_list = model.extract_image_features(sample_list)
                sample_list = model.extract_video_features(sample_list)
                repeat_times = sample_list.image_patch.size(0)
                sample_list.video_patch = sample_list.video_patch.repeat(repeat_times, 1, 1, 1) 
                sample_list = model.transformer.preprocessing_input(sample_list) 
                pooled_output, _ = model.transformer.get_joint_embedding_cross_video2image(
                    sample_list.image_patch,  # [bsz l d], including hard negatives
                    sample_list.video_patch,  # [bsz nf*l d]
                    sample_list.self_attention_mask,  # [bsz l+1], including CLS
                    sample_list.cross_attention_mask,  # [bsz 1 l+1 nf*l]
                    output_attentions=False,
                )
                retrieve_logits = model.transformer.heads["vim"](pooled_output)  # [bsz]
                retrieve_logits_full.append(retrieve_logits)
                imageid_full.extend(imageid_lst)

        _sim = torch.cat(retrieve_logits_full)
        dpath = f"q2g_simmat_{q_videoid}.json"
        dpath = os.path.join(output_root, dpath)
        
        assert len(_sim) == len(imageid_full)
        g_sim = [[k, v.item()] for k,v in zip(imageid_full, _sim)]
        g_sim = [[k,v] for k,v in sorted(g_sim, key=lambda x:x[1], reverse=True)]
        q2g = {q_videoid: g_sim}

        with open(dpath, "w") as f:
            json.dump(q2g, f, indent=1)

    return

if __name__ == "__main__":
    # Configs
    args = parsing_args()
    config_file = args.config_file
    #config_file = 'save/rice_vic_crossvim_rec/config.yaml'
    config = load_yaml(config_file)

    # Output
    exp_name = config.training.experiment_name
    prefix = f"emb_{exp_name}"
    output_root = os.path.join("./results", prefix)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Step1: Calulating query to gallery similarity
    #pairwise_forward(config, output_root, args.device_id)
    
    # Step2: Topk accuracy
    predict_topk(config, output_root, cdt_video=None)
