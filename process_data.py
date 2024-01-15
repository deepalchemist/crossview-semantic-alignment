# -*- coding: utf-8 -*-
# yangwenjie05@kuaishou.com.com @2023-02-18 12:21:01
# Last Change:  2023-11-05 02:26:03

import os
import json
import random
import numpy as np
import pickle as pkl

from tqdm import tqdm

import torch
import torch.nn.functional as F

from cv2_helper import cv2_helper


def check_box():
    #_path = os.path.join("traindata", "_sf_testset_20086_frame2predbox_501656.json")
    #with open(_path, "r") as f:
    #    sf = json.load(f)

    #sample = random.sample(sf.items(), 1)[0]
    #imgpath = sample[0]
    #box_lst = sample[1]
    imgpath = "/data/ad_algo/yangwenjie05/dataset/live_product/trainset/1125/1125_10293757772/frame/10293757772_1669377380386.jpg"
    box_lst = [[181, 355, 394, 491]]
    img = cv2_helper.read_image(imgpath)
    h, w = img.shape[:2]

    #box_lst = [[int(t[0]*w), int(t[1]*h), int(t[2]*w), int(t[3]*h)] for t in box_lst]
    img = cv2_helper.plot_bbox_on_image(box_lst, img=img)
    cv2_helper.write_image(img, "./traindata/debug.jpg")
    return

def mini_trainingset():
    prefix = "other"
    data_root = "./traindata"
    df_training = open(os.path.join(data_root, f"training_{prefix}_mini.txt"), "w")
    df_test = open(os.path.join(data_root, f"test_{prefix}_mini.txt"), "w")

    sf = open(os.path.join(data_root, f"seg_{prefix}.txt")).readlines()
    random.shuffle(sf)

    training_data = sf[:200000]
    test_data = sf[-50000:]
    
    for _line in training_data:
        df_training.write(_line)
    df_training.close()

    for _line in test_data:
        df_test.write(_line)
    df_test.close()
    return

def fmtasr_training():
    _root = "./traindata"
    full_asr = open(os.path.join(_root, "seg_url_asr_4138663.txt"))

    seg2asr = dict()
    for _line in tqdm(full_asr):
        try:
            line = _line.strip().split("\t")
            seg, url, asr = line
            seg2asr[seg] = asr
        except Exception as e:
            print(e)
            print(_line)

    data = open(os.path.join(_root, "_training_data_mini.txt")).readlines()
    df = open(os.path.join(_root, "seg_url_asr_65w_training.txt"), "w")
    for _line in tqdm(data):
        try:
            line = _line.strip().split("\t")
            seg_id, asr_src = line[0],line[-1] 
            asr = seg2asr[seg_id] if seg_id in seg2asr else asr_src
            if "\x01" in asr:
                asr = [t.split()[0] for t in asr.split("\x01")]
                asr = "".join(asr)
                line = f"{seg_id}\t{asr}\n"
            else:
                asr = asr.replace("|", "")
                line = f"{seg_id}\t{asr}\n"
            df.write(line)
        except Exception as e:
            print(e)
            print(_line)
    df.close()
    return

def fmtasr_test():
    _root = "./traindata"
    sf = open("/data/ad_algo/yangwenjie05/dataset/live_product/testset_asr.txt").readlines()
    seg2asr = dict()
    for _line in tqdm(sf):
        line = _line.strip().split(' ')
        seg = "_".join(line[0].split("_")[:-1])
        asr = " ".join(line[1:])
        asr = json.loads(asr)
        if asr is not None:
            asr = [t["text"] for t in asr if t is not None]
            asr = ",".join(asr)
        else:
            asr = "null"
        seg2asr[seg] = asr

    df = open(os.path.join(_root, "seg_asr__test.txt"), "w")
    for seg, asr in seg2asr.items():
        cont = f"{seg}\t{asr}\n"
        df.write(cont)
    df.close()
    return

def queryIteminfo():
    _root = "./traindata/evaldata"
    #item_lst = open(os.path.join(_root, "_training_item_155100.txt")).readlines()
    item_lst = open(os.path.join(_root, "eval_itemid_6108.txt")).readlines()
    item_lst = [t.strip() for t in item_lst]

    batchsize = 100
    split_lst = [item_lst[x:x+batchsize] for x in range(0, len(item_lst), batchsize)]
    
    df = open(os.path.join(_root, "eval_itemTitle_6108.txt"), "w")
    for _split in tqdm(split_lst):
        try:
            item_info = get_item_info_lst(_split)
            for _info in item_info:
                item = _info["item_id"]
                title = _info["title"]
                cate = _info["category_lst"]
                title = ",".join([title]+cate)
                df.write(f"{item}\t{title}\n")
        except Exception as e:
            print(e)
    df.close()
    return

def check_textemb():
    asr = open("./traindata/test_asr_emb_mt_out/0.txt").readlines()
    tit = open("./traindata/test_title_emb_mt_out/total.txt").readlines()

    seg2emb = dict()
    for _line in asr:
        seg, name, emb = _line.strip().split('\t')
        seg2emb[seg] = json.loads(emb)[0]

    item2emb = dict()
    for _line in tit:
        item, emb = _line.strip().split('\t')
        item2emb[item] = json.loads(emb)[0]

    with open("./inputs/clip_to_gtitem.json", "r") as f:
        c2i = json.load(f)
    
    avg_sim = 0
    for _clip, _item in c2i.items():
        emb1 = F.normalize(torch.tensor(seg2emb[_clip]).unsqueeze(0), dim=1)
        emb2 = F.normalize(torch.tensor(item2emb[_item[0]]).unsqueeze(-1), dim=0)
        sim = torch.matmul(emb1, emb2).item()
        avg_sim += sim
    avg_sim = avg_sim / len(c2i)
    print(f"Avarge sim: {avg_sim}")
    return

def count_num_product():
    _path = "./inputs/path_to_mmudetbox.txt"
    sf = open(_path).readlines()
    seg2num = dict()
    for _line in tqdm(sf):
        fpath, box_lst = _line.strip().split("\t")
        box_lst = json.loads(box_lst)
        num_box = len(box_lst)
        seg_id = fpath.split("/")[-2]
        if seg_id not in seg2num:
            seg2num[seg_id] = list()
        seg2num[seg_id].append(num_box)
    seg2num = {k:sum(v)/len(v) for k,v in seg2num.items()}
    
    rst = dict()
    for seg_id, _num in seg2num.items():
        if _num > 7:
            flag = "abundant"
        elif _num <= 3:
            flag = "few"
        else:
            flag = "medium"
        rst[seg_id] = flag

    with open("./inputs/segid_to_numproduct.json", "w") as f:
        json.dump(rst, f, indent=1)
    return

def count_scale():
    #pred_box = "/data/ad_algo/yangwenjie05/dataset/live_product/cache_data/_sf_testset_20086_frame2predbox_501656.json"
    #with open(pred_box, "r") as f:
    #    predbox = json.load(f)
    #gtbox = "/data/ad_algo/yangwenjie05/dataset/live_product/cache_data/testset_20086_frame2gtbox_426751.json"
    #with open(gtbox, "r") as f:
    #    gtbox = json.load(f)
    #src_fpath = open("/data/ad_algo/yangwenjie05/dataset/live_product/cache_data/testset_20086_framename_501656.txt").readlines()
    #rst = dict()
    #for _path in src_fpath:
    #    _path = _path.strip()
    #    _val = gtbox[_path] if _path in gtbox else predbox[_path]
    #    rst[_path] = _val
    #print(len(rst))
    #with open("./inputs/testset_20086_frame2gtbox_501656.json", "w") as f:
    #    json.dump(rst, f, indent=1)
    
    with open("./inputs/testset_20086_frame2gtbox_501656.json", "r") as f:
        gtbox = json.load(f)
    print(len(gtbox))
    MIN = 1000
    MAX = 0
    cnt_zero = 0
    seg2scale = dict()
    for _path, _box in gtbox.items():
        if len(_box)==0:
            cnt_zero += 1
        MIN = min(MIN, len(_box))
        MAX = max(MAX, len(_box))

        area = [(t[2]-t[0])*(t[3]-t[1]) for t in _box]
        try:
            area = sum(area)/len(area) 
        except:
            area = 0.
        seg_id = _path.split("/")[-2]
        if seg_id not in seg2scale:
            seg2scale[seg_id] = list()
        seg2scale[seg_id].append(area)

    print(MIN, MAX, cnt_zero)
    seg2scale = {k:sum(v)/len(v) for k,v in seg2scale.items()}
    rst = dict()
    for _seg, area in seg2scale.items():
        if area<=0.2:
            flag = "small"
        elif area>0.4:
            flag = "large"
        else:
            flag = "medium"
        rst[_seg] = flag

    with open("./inputs/segid_to_scale.json", "w") as f:
        json.dump(rst, f, indent=1)
    print(len(rst))
    return

def count_duration():
    _path = "/data/chenyiyi/projects/dab_git/logs/frame_x7cat2/R50/test_ap_left_ep20/res_test_050.txt"
    sf = open(_path).readlines()
    _path = "/data/chenyiyi/projects/dab_git/logs/frame_x7cat2/R50/test_ap_full_ep20/res_test_050.txt"
    sf += open(_path).readlines()
    print(len(sf))
   
    
    seg2dur = dict()
    for _line in sf:
        _line = _line.strip().split("\t")
        _frame = _line[0]
        box_lst = json.loads(_line[1])
        if "testset_frame_thrsec" in _frame:
            img_path = _frame.replace("testset_frame_thrsec", "testset_frame_20086").split("/")
            img_path = img_path[1:]
            video_name = "_".join(img_path[-2].split("_")[:-1])
            img_name = "_".join(img_path[-1].split("_")[:-2]+img_path[-1].split("_")[-1:])
            dst_imgpath = "/".join(img_path[:6] + [video_name, img_name])
            dst_imgpath = "/" + dst_imgpath
        else:
            dst_imgpath = _frame
        seg_id = dst_imgpath.split("/")[-2]
        if seg_id not in seg2dur:
            seg2dur[seg_id] = list()
        flag = 1 if len(box_lst)>0 else 0
        seg2dur[seg_id].append(flag)
    seg2dur = {k:sum(v)/len(v) for k,v in seg2dur.items()}
    
    rst = dict()
    for _seg, _dur in seg2dur.items():
        if _dur <= 0.4:
            flag = "short"
        elif _dur > 0.7:
            flag = "long"
        else:
            flag = "medium"
        rst[_seg] = flag
    
    with open("./inputs/segid_to_duration.json", "w") as f:
        json.dump(rst, f, indent=1)
    print(len(seg2dur))
    return

def sampling_miniset():
    fpath = "/mnt/csip-113/yangwenjie/dataset/lpr4m/trainset/training_videoinfo_full_4013617.txt"
    sf = open(fpath)
    
    cate2video = dict()
    spu2video = dict()
    for _line in tqdm(sf):
        line = _line.strip().split("\t")
        video_id = line[0]
        cate_id = line[5]
        spu = line[7]
        if cate_id not in cate2video:
            cate2video[cate_id] = list()
        if spu not in spu2video:
            spu2video[spu] = list()
        cate2video[cate_id].append(video_id)
        spu2video[spu].append(video_id)
        #if len(spu2video) == 10:
        #    break

    n_sample = 2
    rst_vid = list()
    for _spu, vid_lst in spu2video.items():
        if len(vid_lst) > n_sample:
            vid_lst = random.sample(vid_lst, n_sample)
        rst_vid += vid_lst
    
    rst_vid = set(rst_vid)
    df = open("./traindata/_mini.txt", "w")
    sf = open(fpath)
    for _line in tqdm(sf):
        line = _line.strip().split("\t")
        video_id = line[0]
        if video_id in rst_vid:
            df.write(_line)
    df.close()
    return

def insert_imagepath():
    sf = open("./traindata/training_videoinfo_full_4013617.txt")
    
    with open('./traindata/frameid_to_path_40135610.pkl', 'rb') as f:
        f2p = pkl.load(f)
    with open('./traindata/imageid_to_path.pkl', 'rb') as f:
        i2p = pkl.load(f)
    
    num_frame = 10
    df = open('./traindata/training_videoinfo_full.txt', 'w')
    for _line in tqdm(sf):
        line = _line.strip().split("\t")
        fid_lst = eval(line[1])
        if len(fid_lst)>num_frame:
            tmp = np.array_split(fid_lst, num_frame)
            tmp = [_t.tolist() for _t in tmp]
            fid_lst = [_t[0] for _t in tmp]

        fpath_lst = [f2p[t] for t in fid_lst if t in f2p]
        line.pop(1)
        line.insert(1, json.dumps(fpath_lst))
        
        imageid = line[3]
        imagepath = i2p[imageid]

        line.insert(4, imagepath)

        df.write('\t'.join(line))
        df.write("\n")

    df.close()
    return

if __name__=="__main__":
    #mini_trainingset()
    #fmtasr_training()
    #fmtasr_test()
    #check_box()
    #queryIteminfo()
    #check_textemb()
    #count_num_product()
    #count_scale()
    #count_duration()
    #sampling_miniset()
    insert_imagepath()
