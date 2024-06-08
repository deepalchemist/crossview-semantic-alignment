# -*- coding: utf-8 -*-
# hokkien.ywj@gmail.com @2024-05-03 01:12:45
# Last Change:  2024-05-06 07:57:03

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
from mmf.datasets.builders.movingfashion.dataset import MovingFashionDataset

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config_file', required=True, type=str, help='path/to/config/file')
    parser.add_argument('-d', '--device_id', type=int, default=0)
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

            if cur_pred == gt[q_name]["gt"]:
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
    
    data_root = config.dataset_config.movingfashion.data_dir
    gt_path = os.path.join(data_root, "annotations/video_to_gt.json")
    with open(gt_path, "r") as f:
        gt = json.load(f)

    # Conditional 
    if cdt_video is not None:
        gt = {k:v for k,v in gt.items() if k in cdt_video}

    #gt = {k:v for k,v in gt.items() if k in pred}
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
    #ckpt_path = os.path.join(config.env.save_dir, f"iter2000.ckpt")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    #state_dict = state_dict["model"]
    fmt_state_dict = OrderedDict()
    for _key, _val in state_dict.items():
        if _key.startswith("module."):
            _key = _key.replace("module.", "")
        fmt_state_dict[_key] = _val
    model.load_state_dict(fmt_state_dict, strict=True)
    print(f"Successfully load checkpoint from: {ckpt_path}")
    model.eval()
    return model

def extract_gallery(config, output_root, device_id):
    BATCH_SIZE = 32
    device = torch.device(f"cuda:{device_id}")

    # Init model
    model = init_model(config)
    model = model.to(device)
    
    # Gallery set
    gallery_set = MovingFashionDataset(
        config.dataset_config.movingfashion, 
        dataset_type="test",
        index=1,
        )
    gallery_set.init_processors()
    gallery_loader= DataLoader(
            gallery_set,
            batch_size=BATCH_SIZE,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            )


    l2g = dict()
    _cnt = -1
    for sample in tqdm(gallery_loader, desc="gallery forward"):
        _cnt += 1
        #if _cnt == 5: break
        sample = Sample(sample)
        imageid_lst = sample.image_id

        # Inference
        with torch.no_grad():
            sample.image = sample.image.to(device)
            output_dict = model(sample)
        img_embs = output_dict["targets"]  # [bs d], l2 normalized and multiply a factor

        assert len(imageid_lst)==img_embs.size(0)
        for _imageid, _emb in zip(imageid_lst, img_embs):
            if "movingfashion" not in l2g:
                l2g["movingfashion"] = {"name": list(), "emb": list()}
            l2g["movingfashion"]["name"].append(_imageid)
            l2g["movingfashion"]["emb"].append(_emb)

    l2g["movingfashion"]["emb"] = torch.stack(l2g["movingfashion"]["emb"], dim=0)

    with open(os.path.join(output_root,"gallery_feature.pkl"), "wb") as f:
        pkl.dump(l2g, f)
    
    model = model.cpu()
    del model
    torch.cuda.empty_cache() 
    return

def extract_query(config, output_root, device_id):
    BATCH_SIZE = 32
    device = torch.device(f"cuda:{device_id}")
    with open(os.path.join(output_root, "gallery_feature.pkl"), "rb") as f:
        live2g = pkl.load(f)

    live2emb = {k:v["emb"].to(device) for k,v in live2g.items()}
    for k in live2emb.keys():
        live2g[k]["emb"]=live2emb[k] 
 
    # Init model
    model = init_model(config)
    model = model.to(device)
 
    # Init dataloader
    query_set = MovingFashionDataset(
            config.dataset_config.movingfashion, 
            dataset_type="test",
            index=0,
            )
    query_set.init_processors()
    query_loader = DataLoader(
        query_set,
        batch_size=BATCH_SIZE,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        )
    
    _cnt = -1
    for sample in tqdm(query_loader, desc=f"query forward"):
        # video: (batch_size max_frame 3 h w)
        # video_temporal_mask: (batch_size max_frame)
        _cnt += 1
        #if _cnt == 5: break
        sample = Sample(sample)
        videoid_lst = sample.video_id

        with torch.no_grad():
            sample.video = sample.video.to(device)
            sample.video_temporal_mask = sample.video_temporal_mask.to(device)
            output_dict = model(sample)
        
        video_embs = output_dict["scores"]  # [bs d], l2 normalized and multiply a factor
        assert len(videoid_lst)==video_embs.size(0)

        for _videoid, q_f in zip(videoid_lst, video_embs):
            g_f = live2g["movingfashion"]["emb"]  # gallery image feature, [m d]
            #g_f = g_f / g_f.norm(dim=-1, keepdim=True)

            q_f = q_f[None, :]
            #q_f = q_f/ q_f.norm(dim=-1, keepdim=True)

            _sim = torch.matmul(q_f, g_f.t()).squeeze(0) # (1, full_gallery)

            dpath = f"q2g_simmat_{_videoid}.json"
            dpath = os.path.join(output_root, dpath)

            item_onshelf = live2g["movingfashion"]["name"]
            g_sim = [[k, v.item()] for k,v in zip(item_onshelf, _sim)]
            g_sim = [[k,v] for k,v in sorted(g_sim, key=lambda x:x[1], reverse=True)]
            q2g = {_videoid: g_sim}

            with open(dpath, "w") as f:
                json.dump(q2g, f, indent=1)
    return

if __name__ == "__main__":
    # Configs
    args = parsing_args()
    config_file = args.config_file
    #config_file = 'save/mf_rice_vic_vim_rec/config.yaml'
    config = load_yaml(config_file)

    # Output
    exp_name = config.training.experiment_name
    prefix = f"emb_{exp_name}"
    output_root = os.path.join("./results", prefix)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Step1: Exacting gallery feature
    extract_gallery(config, output_root, args.device_id)

    # Step2: Calulating query to gallery similarity
    extract_query(config, output_root, args.device_id)
    
    # Step3: Topk accuracy
    predict_topk(config, output_root, cdt_video=None)

