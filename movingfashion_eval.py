
# -*- coding: utf-8 -*-
# hokkien.ywj@gmail.com @2022-09-16 14:58:32
# Last Change:  2023-08-15 11:01:32

import os
import sys
import math
import time
import json
import argparse
import numpy as np
try:
    import pickle5 as pkl
except:
    import pickle as pkl

from PIL import Image
from tqdm import tqdm
import multiprocessing
from collections import defaultdict

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F

from module.modeling import CLIP4Clip
from module.util_module import genPatchmask, IOU
from dataset.rawframe_util import RawVideoExtractor
image_resolution = 224
rawVideoExtractor = RawVideoExtractor(centercrop=True, size=(image_resolution, image_resolution))  # (12, 7)
rawImageExtractor = RawVideoExtractor(centercrop=True, size=(image_resolution, image_resolution))  # (12, 7)
 
 # Generate grid
patchsize = (32, 32)
h, w = patchsize
grid_h = image_resolution // h
grid_w = image_resolution // w
GRIDS = []
for i in range(grid_h):
    for j in range(grid_w):
        ltrb = [j*w, i*h, (j+1)*w, (i+1)*h]    
        GRIDS.append(ltrb)   

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default="./ckpt/lpr4m_loosetype/pytorch_model.bin.26",
                        help='The trained checkpoint')
    parser.add_argument('--sim_header', type=str,
                        default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help='one-stage or two-stage')
    parser.add_argument('--cross_num_hidden_layers', type=int,
                        default=2,
                        help='layer number of the second stage')
    parser.add_argument('--loose_type', action='store_false', help='if one-stage or not')
    parser.add_argument('--recons_feat', action='store_true', help='')
    parser.add_argument('--ipd', action='store_true', help='')
    parser.add_argument('--embedding_sim',
                        action='store_false',
                        help='using embedding to compute similarity?')
    parser.add_argument('--mode', type=str,
                        default="video",
                        choices=["video", "frame"],
                        help='query is video or frame?')
    args = parser.parse_args()
    return args

def _get_rawvideo(framepath_lst, choice_video_ids, rawVideoExtractor, max_frames=10, slice_framepos=2, frame_order=0):
    num_video = len(choice_video_ids)
    video_mask = np.zeros((num_video, max_frames), dtype=np.int64)
    max_video_length = [0] * num_video

    # Pair x L x T x 3 x H x W
    video = np.zeros((num_video, max_frames, 1, 3,
                      rawVideoExtractor.size_h, rawVideoExtractor.size_w), dtype=np.float64)

    for i, video_id in enumerate(choice_video_ids):
        raw_video_data = rawVideoExtractor.get_video_data(framepath_lst[i])
        raw_video_data = raw_video_data['video']

        if len(raw_video_data.shape) > 3:
            raw_video_data_clip = raw_video_data
            # L x T x 3 x H x W
            raw_video_slice = rawVideoExtractor.process_raw_data(raw_video_data_clip)
            if max_frames < raw_video_slice.shape[0]:
                if slice_framepos == 0:
                    video_slice = raw_video_slice[:max_frames, ...]
                elif slice_framepos == 1:
                    video_slice = raw_video_slice[-max_frames:, ...]
                else:
                    sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice

            video_slice = rawVideoExtractor.process_frame_order(video_slice, frame_order=frame_order)

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

    video = torch.tensor(video)
    video_mask = torch.tensor(video_mask)
    return video, video_mask
    
def check_image_file(filename):
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])

def most_freq(List):
    return max(set(List), key = List.count)

def mergeSubdict(data_root):
    dstdir = data_root
    liveid_lst = [t for t in os.listdir(dstdir) if t.isnumeric()]
    for liveid in liveid_lst:
        dict_dir = os.path.join(dstdir, liveid)
        dict_lst = [_t for _t in os.listdir(dict_dir) if _t.startswith("frame2item_") and "final" not in _t]
        fpath_lst = [os.path.join(dict_dir, _t) for _t in dict_lst]

        rst = dict()
        for _path in fpath_lst:
            with open(_path, "r") as f:
                data = json.load(f)
            for _key, _val in data.items():
                new_key = "_".join([liveid, _key])
                if new_key not in rst:
                    rst[new_key] = list()
                rst[new_key] += _val
        with open(os.path.join(dict_dir, "_frame2item.json"), "w") as f:
            json.dump(rst, f, indent=1)
    return 

def fmtGroundTruth(data_root):

    with open(f"{data_root}/test.json", "r") as f:
        testset = json.load(f)
    
    clip2gt = dict()
    df = open(os.path.join(data_root, "clipid_videopath_imagepath_srcid.txt"), "w")
    for _cid, _val in tqdm(testset.items(), desc="clip forward"):
        assert len(_val["video_paths"])==1
        video_path = _val["video_paths"][0]
        _cid = os.path.basename(video_path).split(".")[0]
        image_path = _val["img_path"]
        srcid = _val["source"]
        _cont = f"{_cid}\t{video_path}\t{image_path}\t{srcid}\n"
        df.write(_cont)
        if _cid not in clip2gt:
            clip2gt[_cid] = dict()
        clip2gt[_cid]["gt"] = os.path.basename(image_path).split(".")[0]
        clip2gt[_cid]["source"] = srcid
    df.close()
    with open(f"{data_root}/clip2gt.json", "w") as f:
        json.dump(clip2gt, f, indent=1)
    return

def init_model(sim_header, cross_num_hidden_layers, loose_type, device, ckpt_path=None, training=True, recons_feat=False, embedding_sim=True):
    if ckpt_path is not None:
        model_state_dict = torch.load(ckpt_path, map_location=device)  # torch.device("cpu") torch.device("cuda:0")
    else:
        model_state_dict = None
        
    model = CLIP4Clip.from_pretrained(cross_model_name="cross-base", 
            max_words=32, max_frames=10, linear_patch="2d", pretrained_clip_name="ViT-B/32", 
            loose_type=loose_type, sim_header=sim_header, cross_num_hidden_layers=cross_num_hidden_layers,
            state_dict=model_state_dict, cache_dir=None, type_vocab_size=2, training=training, recons_feat=recons_feat, embedding_sim=embedding_sim)
    
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    return model

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

def topkPrediction(input_root, data_root, prefix="video_meanP"):
    #dst_root = os.path.join(data_root, prefix)
    dst_root = data_root
    clip_lst = [t for t in os.listdir(dst_root) if t.startswith("q2g_simmat_")]
    
    # Merge 
    pred_path = os.path.join(dst_root, f"_{prefix}_q2topk.json")
    if not os.path.isfile(pred_path):
        rst = dict()
        for clip_name in clip_lst:
            dstpath = os.path.join(dst_root, clip_name)
            with open(dstpath, "r") as f:
                q2g = json.load(f)
            for _key, _val in q2g.items():
                #_val = [[t[0].split("_")[0], t[1]] for t in _val]
                rst[_key] = _val
        print(f"Total number of clip: {len(rst)}")
        
        with open(pred_path, "w") as f:
            json.dump(rst, f, indent=1)

    # Calculate topk hit and acc
    with open(pred_path, "r") as f:
        pred = json.load(f)
    
    with open(os.path.join(input_root, "clip2gt.json"), "r") as f:
        gt = json.load(f)
    
    gt = {k:v for k,v in gt.items() if k in pred}
    #gt = {k:v for k,v in gt.items() if v["source"]==1}
    #pred = {k:v for k,v in pred.items() if k in gt}
    print(f"Total number of gt: {len(gt)}")
    
    calculate_acc(pred, gt, top_k=10, root_dir=None, write_dir=None, output_badcase=False)
    return

def clipWorker(subset, process_index, data_root, 
        embedding_sim, recons_feat, ipd, sim_header, 
        cross_num_hidden_layers, loose_type, ckpt_path, 
        out_root, mode, n_gpu, num_frame=10):

    model_name = ckpt_path.split("/")[1]
    sim_type = "emb" if embedding_sim else "decoder"

    sf = open(f"{data_root}/clipid_videopath_imagepath_srcid.txt").readlines()
    imgpath_lst = [_t.strip().split("\t")[2] for _t in sf]
    
    #with open("./traindata/_sf_testset_20086_frame2predbox_501656.json", "r") as f:
    #    f2box = json.load(f)
    device_id = process_index % n_gpu
    device = torch.device(f"cuda:{device_id}")
    model = init_model(sim_header, cross_num_hidden_layers, loose_type, device, ckpt_path=ckpt_path, training=False, recons_feat=recons_feat, embedding_sim=embedding_sim)
    model = model.to(device)


    # >>> Extract shop image embedding
    item_onshelf = [os.path.join(data_root, _t) for _t in imgpath_lst]
    item_onshelf.sort()
    # Collect batch of item on shelf
    image_batchsize = 512 if embedding_sim else video_batchsize
    
    item_batch_lst = [item_onshelf[t:t+image_batchsize] for t in range(0,len(item_onshelf),image_batchsize)]
    index_lst = list(range(len(item_batch_lst)))
    
    image_output_lst = list()
    image_hidden_lst = list()
    for _index, item_path_lst in enumerate(item_batch_lst):
        #if _index == 2:
        #    break
        image_data = [rawImageExtractor.get_video_data([t])["video"] for t in item_path_lst]
        image_data = torch.cat(image_data, dim=0)  # (bs 3 H W)
        
        item_emb = None
        
        # Image inference
        with torch.no_grad():
            image_data = image_data.to(device)
            image_batchsize = image_data.size(0)
            image_output, image_hidden = model.get_visual_output(image_data, image_batchsize, attn_mask=None)
        image_output_lst.append(image_output)  # (bs 1 dim_feat)
        image_hidden_lst.append(image_hidden)
    image_output = torch.cat(image_output_lst, dim=0)
    image_hidden = torch.cat(image_hidden_lst, dim=0)
    # >>>
    
    cnt = -1
    for clipid in tqdm(subset, desc=f"process {process_index}-th forward"):
        cnt += 1
        #if cnt==2:
        #    break
        try:
            dpath = f"q2g_simmat_{clipid}.json"
            dpath = os.path.join(out_root, dpath)
            q2g = dict()
            asr_emb = None

            frame_path = os.path.join(data_root,"frames",clipid)
            assert os.path.exists(frame_path), f"{frame_path} does not exists!"
            frame_lst = [t for t in os.listdir(frame_path) if t.endswith(".png")]
            #frame_lst = [os.path.join(frame_path, _t) for _t in frame_lst]
            frame_lst.sort()

                
            # Predict an item for video or frame?
            assert mode=="video"
            if len(frame_lst) > num_frame:
                sample_index = np.linspace(0, len(frame_lst), num=num_frame, endpoint=False, dtype=int)
                video_tictoc = [frame_lst[t] for t in sample_index]
            else:
                video_tictoc = frame_lst
                
            query2item = defaultdict(list)
            batch_photo_lst = [video_tictoc]
                
            # Collecte batch photo
            photo_lst = list()
            photomask_lst = list()
            patchmask_lst = list()
            # Go through live clips
            for clip_framepath_lst in batch_photo_lst:
                frame_path_lst = [os.path.join(frame_path, t) for t in clip_framepath_lst]
                video, video_mask = _get_rawvideo([frame_path_lst], ["random_clipid"], rawVideoExtractor)  # video_mask (1 max_frame)

                # Opt1: Patch mask
                if ipd:
                    box_lst = [f2box[t] for t in frame_path_lst]
                    box_lst = [[[t[0]*image_resolution,t[1]*image_resolution,t[2]*image_resolution,t[3]*image_resolution] for t in _box] for _box in box_lst]
                    patch_mask = [genPatchmask(t, GRIDS, iou_thresh=0.02) for t in box_lst]
                    if len(patch_mask)<video_mask.size(1):
                        num_pad = video_mask.size(1)-len(patch_mask)
                        patch_mask = patch_mask + [patch_mask[-1]]*num_pad
                    patch_mask = torch.tensor(patch_mask, dtype=torch.long)  # (num_frame 1+num_patch)
                    #assert patch_mask.size(0)==video_mask.size(1)
                    #patch_mask = patch_mask*video_mask.squeeze(0).unsqueeze(-1)

                # Opt2: Patch mask
                else:
                    patch_mask = video_mask.squeeze(0).unsqueeze(-1).repeat(1, 1+49)  # (num_frame 1+num_patch)

                photo_lst.append(video)
                photomask_lst.append(video_mask)
                patchmask_lst.append(patch_mask)
                
            video_batchsize = len(photo_lst)    
            video_data = torch.cat(photo_lst, dim=0).squeeze(2) # (bs max_frame 1 3 H W) to (bs max_frame 3 H W)
            video_data = video_data.view(-1, *video_data.shape[2:])  # (bs*max_frame 3 H W)
            video_mask = torch.cat(photomask_lst, dim=0)  # (bs max_frame)
            patch_mask = torch.stack(patchmask_lst, dim=0)  # (bs max_frame 1+num_patch)
            
            # Video inference
            with torch.no_grad():
                video_data = video_data.to(device)
                video_mask = video_mask.to(device)
                patch_mask = patch_mask.to(device)

                # Opt1
                if ipd:
                    video_attn_mask = patch_mask.view(-1, patch_mask.size(-1)).contiguous()
                    video_attn_mask = video_attn_mask==0
                # Opt2
                else:
                    video_attn_mask = None

                video_output, video_hidden = model.get_visual_output(video_data, video_batchsize, attn_mask=video_attn_mask)
                video_hidden = video_hidden.view(video_batchsize, -1, *video_hidden.shape[-2:])

            # Collect batch of item on shelf
            #image_batchsize = 512 if embedding_sim else video_batchsize
            
            #item_batch_lst = [item_onshelf[t:t+image_batchsize] for t in range(0,len(item_onshelf),image_batchsize)]
            #index_lst = list(range(len(item_batch_lst)))
    
            #sim_lst = list()
            #for item_path_lst in item_batch_lst:
            #    image_data = [rawImageExtractor.get_video_data([t])["video"] for t in item_path_lst]
            #    image_data = torch.cat(image_data, dim=0)  # (bs 3 H W)
            #    
            #    item_emb = None
            #    
            #    # Image inference
            #    with torch.no_grad():
            #        image_data = image_data.to(device)
            #        image_batchsize = image_data.size(0)
            #        image_output, image_hidden = model.get_visual_output(image_data, image_batchsize, attn_mask=None)
            #        
            #        output = model.get_similarity_logits(
            #            image_output, image_hidden, 
            #            video_output, video_hidden, video_mask, patch_mask, 
            #            asr_emb, item_emb,
            #            shaped=True, loose_type=True,  # TODO_BUG
            #        )

            #        
            #        if embedding_sim:
            #            i2v_simmat = output["contrastive_logit"]
            #        else:
            #            i2v_simmat = output["pairwise_logit"]
            #        v2i_simmat = i2v_simmat.T
            #    sim_lst.append(v2i_simmat)
            #sim = torch.cat(sim_lst, dim=1)  # (bs_query, full_gallery)

            output = model.get_similarity_logits(
                        image_output, image_hidden, 
                        video_output, video_hidden, video_mask, patch_mask, 
                        asr_emb, item_emb,
                        shaped=True, loose_type=True,  # TODO_BUG
                    )
            if embedding_sim:
                i2v_simmat = output["contrastive_logit"]
            else:
                i2v_simmat = output["pairwise_logit"]
            sim = i2v_simmat.T

            assert sim.size(1) == len(item_onshelf)
            
            assert mode == "video"
            assert video_batchsize==1 and sim.size(0)==1
            #q_name = "_".join(_batch[0][0].split("_")[:-1])
            q_name = clipid
            g_sim = [[os.path.basename(k).split(".")[0], v.item()] for k,v in zip(item_onshelf, sim.squeeze(0))]
            g_sim = [[k,v] for k,v in sorted(g_sim, key=lambda x:x[1], reverse=True)]
            q2g.update({q_name: g_sim})
                                
            with open(dpath, "w") as f:
                json.dump(q2g, f, indent=1)
        except Exception as e:
            print(f"Meeting error, clip: {clipid}, {e}")
    return

def mpProcessClip(data_root, out_root, mode, n_gpu,
         ckpt_path, sim_header, loose_type, cross_num_hidden_layers, embedding_sim, recons_feat, ipd):
    sf = open(f"{data_root}/clipid_videopath_imagepath_srcid.txt").readlines()
    clipid_lst = [_t.strip().split("\t")[0] for _t in sf]
    print(f"Total number of clip: {len(clipid_lst)}")
    
    n_process = n_gpu*1 if mode=="video" else n_gpu*6
    n_per_process = math.ceil(len(clipid_lst)/n_process)
    input_split = [clipid_lst[i:i+n_per_process] for i in range(0,len(clipid_lst),n_per_process)]
    index_lst = list(range(len(input_split)))
    
    clipWorker(input_split[0], index_lst[0], data_root, args.embedding_sim, args.recons_feat, args.ipd, args.sim_header, args.cross_num_hidden_layers, args.loose_type, args.ckpt_path, out_root, args.mode, n_gpu)
    #import pdb; pdb.set_trace()

    # Multiprocessing 
    #jobs = []
    #ctx = multiprocessing.get_context("spawn")
    #for _index in index_lst:
    #    sub_batch = input_split[_index]
    #    p = ctx.Process(target=clipWorker, args=(sub_batch, _index, data_root, embedding_sim, recons_feat, ipd,
    #                                             sim_header, cross_num_hidden_layers, loose_type, ckpt_path, 
    #                                             out_root, mode, n_gpu
    #                                             ))
    #    p.start()
    #    jobs.append(p)
    #for p in tqdm(jobs, desc="join subprocess"):
    #    p.join()
    return

if __name__ == "__main__":
    args = parsing_args()

    data_root = "/mnt/csip-113/yangwenjie/dataset/movingfashion"

    model_name = args.ckpt_path.split("/")[2]
    sim_type = "emb" if args.embedding_sim else "decoder"
    #prefix = f"{args.mode}_{sim_type}_{model_name}_cattext"
    prefix = f"{args.mode}_{sim_type}_{model_name}"
    out_root = os.path.join("./outputs", prefix)
    if not os.path.exists(out_root):
        os.makedirs(out_root)
        
    n_gpu = 1
    
    # Formatting key-value ground-truth 
    #fmtGroundTruth(data_root)
    
    # Step1: Calulating query to gallery similarity
    #mpProcessClip(data_root, out_root, args.mode, n_gpu, args.ckpt_path, args.sim_header, args.loose_type, args.cross_num_hidden_layers, args.embedding_sim, args.recons_feat, args.ipd)
    
    # Step2: 
    topkPrediction(data_root, out_root, prefix)
    
    # Step3:
    #pr_curve(data_root)

    #count_query_gallery()
    #collect_frame_bbox()
