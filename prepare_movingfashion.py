# -*- coding: utf-8 -*-
# hokkien.ywj@gmail.com @2024-04-01 02:33:38
# Last Change:  2024-04-08 08:20:03

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='moving fashion data root')
    args = parser.parse_args()
    return args

def extract_frames(data_root):
    ''' Extract a frame per second 
    '''
    video_root = os.path.join(data_root, "videos")
    video_lst = [t for t in os.listdir(video_root) if t.endswith(".mp4")]
    for _vid in tqdm(video_lst, desc="extract video frame"):
        src_path = os.path.join(video_root, _vid)
        dst_root = os.path.join(data_root, "frames", _vid)
        Path(dst_root).mkdir(parents=True, exist_ok=True)
        cmd = f"ffmpeg -i {src_path} -vf fps=1 {dst_root}/{_vid}_%04d.png"
        os.system(cmd)
    return

def formatting_groundtruth(data_root):
    NUM_FRAME = 10
    with open(f"{data_root}/test.json", "r") as f:
        testset = json.load(f)
    dst_root = os.path.join(data_root, "annotations") 
    Path(dst_root).mkdir(parents=True, exist_ok=True)

    video2gt = dict()
    df = open(os.path.join(dst_root, "videoid_videopath_imagepath_srcid.txt"), "w")
    g_file = open(os.path.join(dst_root, "gallery_full.txt"), "w")
    q2info = dict()
    for _cid, _val in tqdm(testset.items(), desc="video forward"):
        assert len(_val["video_paths"])==1
        video_path = _val["video_paths"][0]
        _cid = os.path.basename(video_path).split(".")[0]
        image_path = _val["img_path"]
        srcid = _val["source"]
        _cont = f"{_cid}\t{video_path}\t{image_path}\t{srcid}\n"
        df.write(_cont)
        g_cont = f"{image_path}\n"
        g_file.write(g_cont)

        frame_dir = os.path.join(data_root, "frames", _cid)
        if not os.path.isdir(frame_dir):
            print(f"{frame_dir}")
            continue
        
        frame_name = [t for t in os.listdir(frame_dir) if t.endswith(".png")]
        frame_path = [os.path.join("frames", _cid, t) for t in frame_name]
        frame_path.sort()
        if len(frame_path) == 0:
            print(f"{frame_dir}")
            continue

        if _cid not in video2gt:
            video2gt[_cid] = dict()
        video2gt[_cid]["gt"] = os.path.basename(image_path).split(".")[0]
        video2gt[_cid]["source"] = srcid

        if len(frame_path) > NUM_FRAME:
            sample_index = np.linspace(0, len(frame_path), num=NUM_FRAME, endpoint=False, dtype=int)
            tgt_frames = [frame_path[t] for t in sample_index]
        else:
            tgt_frames = frame_path

        if _cid not in q2info:
            q2info[_cid] = dict()
        q2info[_cid]["frames"] = tgt_frames
        q2info[_cid]["gt"] = video2gt[_cid]["gt"]


    df.close()
    g_file.close()
    with open(f"{dst_root}/video_to_gt.json", "w") as f:
        json.dump(video2gt, f, indent=1)
    
    with open(f"{dst_root}/query_tenframes.json", "w") as f:
        json.dump(q2info, f, indent=1)
    return

def formatting_training_file(data_root):
    with open(f"{data_root}/train.json", "r") as f:
        trainset = json.load(f)

    dst_root = os.path.join(data_root, "annotations") 
    Path(dst_root).mkdir(parents=True, exist_ok=True)

    NUM_FRAME = 10
    df = open(os.path.join(dst_root, "training_videoinfo_full.txt"), "w")
    for _imgid, _val in tqdm(trainset.items(), desc="forward"):
        img_path = _val["img_path"]
        _imgid = os.path.basename(img_path).split(".")[0]
        src_id = _val["source"]
        videopath_lst = _val["video_paths"]
        for _vidpath in videopath_lst:
            video_id = os.path.basename(_vidpath).replace(".mp4", "")
            frame_root = os.path.join(data_root, "frames", video_id)
            if not os.path.exists(frame_root):
                print(f"! Bad root: {frame_root} from video path {_vidpath}")
                continue
            frame_lst = [t for t in os.listdir(frame_root) if t.endswith(".png")]
            if len(frame_lst) == 0:
                print(f"! Bad root: {frame_root} from video path {_vidpath}")
                continue
            frame_lst.sort()
            if len(frame_lst) > NUM_FRAME:
                sample_index = np.linspace(0, len(frame_lst), num=NUM_FRAME, endpoint=False, dtype=int)
                tgt_frames = [frame_lst[t] for t in sample_index]
            else:
                tgt_frames = frame_lst

            frame_lst = [os.path.join("frames", video_id, t) for t in tgt_frames]
            frame_lst = json.dumps(frame_lst)
            
            # Write a row
            _cont = f"{video_id}\t{frame_lst}\t{_imgid}\t{img_path}\t{src_id}\n"
            df.write(_cont)
 
    df.close()
    return

if __name__=="__main__":
    args = parsing_args()
    
    # Extract video frames
    extract_frames(args.data_root)

    # Formatting key-value ground-truth 
    formatting_groundtruth(args.data_root)

    # Formatting training file
    formatting_training_file(args.data_root)
 
