

# Last Change:  2023-12-17 03:25:31

import os
import torch
from torch.utils.data import DataLoader
from dataset.getitem_lpr4m import LPR4M_DataLoader, LPR4M_TrainDataLoader
from dataset.getitem_msrvtt import MSRVTT_DataLoader, MSRVTT_TrainDataLoader

def dataloader_lpr4m_train(args, tokenizer):
    lpr4m_dataset = LPR4M_TrainDataLoader(
        data_root=args.data_root,
        data_path="./traindata/training_videoinfo_full.txt",
        #data_path="./traindata/training_videoinfo_mini.txt",
        #data_path="./traindata/training_videoinfo_debug.txt",
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lpr4m_dataset)
    dataloader = DataLoader(
        lpr4m_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lpr4m_dataset), train_sampler

def dataloader_lpr4m_test(args, tokenizer, subset="test"):
    lpr4m_testset = LPR4M_DataLoader(
        data_root=args.data_root,
        #data_path="training_videoinfo.txt",
        #data_path="training_videoinfo_mini.txt",
        data_path="training_videoinfo_debug.txt",
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader = DataLoader(
        lpr4m_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(lpr4m_testset)

def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        data_root=args.data_root,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        data_root=args.data_root,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

DATALOADER_DICT = {}
DATALOADER_DICT["lpr4m"] = {"train":dataloader_lpr4m_train, "val":dataloader_lpr4m_test, "test":None}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test, "test":None}
