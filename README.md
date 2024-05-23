<div align="center">

# Cross-view Semantic Alignment for Livestreaming Product Recognition

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://mmf.sh/"><img alt="MMF" src="https://img.shields.io/badge/MMF-0054a6?logo=meta&logoColor=white"></a>
[![Conference](https://img.shields.io/badge/ICCV-2023-6790AC.svg)](https://iccv2023.thecvf.com/)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2308.04912-B31B1B.svg)](https://arxiv.org/pdf/2308.04912.pdf)

</div>

The official code of ICCV 2023 paper
[Cross-view Semantic Alignment for Livestreaming Product Recognition](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Cross-view_Semantic_Alignment_for_Livestreaming_Product_Recognition_ICCV_2023_paper.html)

This work makes two contributions: :grin: (1) collecting a large-scale and diverse dataset for the video-to-shop task, which aims to match products (i.e., clothes) in videos to online shopping images. :grin: (2) proposing a novel pre-training framework to exploit the specialties of e-commerce data.

In the original paper, we reported the performance of the model trained on the complete dataset of 4 million samples using 8 V100 GPUs. However, we had concerns about the accessibility of the dataset and the reproducibility of the model performances. The reasons are as follows: :warning: (1) the large size of 4 million samples makes the data difficult to download (time-consuming and error-prone) and requires large storage space (4 TB). :warning: (2) using 8 V100 GPUs sets high resource requirements for model training. Considering these factors, we made the following adjustments in this repository: :grin: (1) we reduced the number of training samples. We sampled a training subset containing ***750K*** samples from the full 4 million samples, which only reduced the number of samples of the head categories without sampling the tail categories (long-tail). :grin: (2) we used the more widely available ***RTX 3090 GPU*** instead of ***V100 GPU*** to train the model (without sacrificing accuracy), facilitating the reproduction of model results.

## Architecture
<p align="center">
  <img width="600" height="400" src="./images/model.png">
</p>

## Installation
Follow installation instructions in the [documentation](https://mmf.sh/docs/),
or refer to ***requirements.txt***.

## Documentation

Learn more about MMF [here](https://mmf.sh/docs).

## Data Preparation
### Prepare LPR750K
Download the sampled subset with 750k video-image pairs (named LPR750K).
```bash
bash scripts/download_lpr750k.sh
```
The LPR750K dataset takes about 370GB.
```bash
|--lpr750k
    |--annotations
    |    |--training_videoinfo_750k.txt
    |    |--test_videoid_to_gtimage_20079.json
    |    |--...
    |--query
    |    |--query_video_1
    |    |--query_video_2
    |    |--...
    |--gallery
    |    |--livestreaming_id_1
    |    |    |--gallery_image_1
    |    |    |--gallery_image_2
    |    |    |--...
    |    |--livestreaming_id_2
    |    |    |--gallery_image_k
    |    |    |--gallery_image_m
    |    |    |--...
    |    |--...
    |--training_image
    |--training_video_00
    |--training_video_01
    |--...
    |--training_video_07
```

### Prepare MovingFashion
The dataset can be downloaded from the [official code repository](https://github.com/HumaticsLAB/SEAM-Match-RCNN) (about 24GB)

```bash
|--movingfashion
    |--videos
    |    |--xxx.mp4
    |    |--xxx.mp4
    |    |--...
    |--imgs
    |    |--xxx.jpg
    |    |--xxx.jpg
    |    |--...
    |--test.json
    |--train.json
```
Then prepare movingfashion dataset.
```bash
python prepare_movingfashion.py --data_root /path/to/movingfashion/dataset/
```

## Evaluation
We perform ablation study on LPR4M and compare the proposed method with SOTA on LPR4M and MovingFashion (the results on WAB dataset is coming soon).

### Ablation study 
<table align="center">
    <thead>
        <tr>
            <th colspan=3></th>
            <th colspan=2>LPR4M</th>
            <th colspan=2>MovingFashion</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ICL</td>
            <td>PMD</td>
            <td>PFR</td>
            <td>R1</td>
            <td>Checkpoints</td>
            <td>R1</td>
            <td>Checkpoints</td>
        </tr>
        <tr>
          <td>&#10004</td>
          <td></td>
          <td></td>
          <td>31.50</td>
          <td><a href="https://drive.google.com/file/d/1DKJRDzsYAih_LBe2eTaeIF8hK3-J6rnU/view?usp=drive_link">Google Drive</a></td>
          <td>74.76</td>
          <td><a href="https://drive.google.com/file/d/14J_-Zf4UZ8j-PSzWeOjqeSblt62GtzU6/view?usp=drive_link">Google Drive</a></td>
        </tr>
        <tr>
          <td>&#10004</td>
          <td>&#10004</td>
          <td></td>
          <td>xx.xx</td>
          <td><a href="">Google Drive</a></td>
          <td>76.87</td>
          <td><a href="https://drive.google.com/file/d/1rqcL8rp6djdQcb6zaSx6N9_FEyfWNtnP/view?usp=drive_link">Google Drive</a></td>
        </tr>
        <tr>
          <td>&#10004</td>
          <td>&#10004</td>
          <td>&#10004</td>
          <td>xx.xx</td>
          <td><a href="">Google Drive</a></td>
          <td>77.92</td>
          <td><a href="https://drive.google.com/file/d/1ZdCX1fkceqjhAN0pC6IVvzAsIOWwjUD3/view?usp=drive_link">Google Drive</a></td>
        </tr>
    </tbody>
</table>
 
:star2: On LPR4M, the models are train via Multi-Node Distributed Data Parallel (DDP). We use 2 nodes and each node has 2 x RTX 3090 GPUs with 24GB of memory each and a global batch size of 128. For other configuration, please refer to the yaml file, i.e., `mmf/projects/videotoshop/configs.e2e_pretraining_xxx.yaml`. 

Firstly create dir `save/mf_rice_vic`, then download the config files **config.yaml** from [Google Drive](https://drive.google.com/drive/folders/1ynewedJx104xAaiw42z1vHx7H2CZ0urZ?usp=drive_link) and download the trained models **rice_final.pth** from the links in the table above.
```bash
|--save
    |--mf_rice_vic
    |    |--config.yaml
    |    |--rice_final.pth
    |    |--...
    |--mf_rice_vic_vim
    |    |--config.yaml
    |    |--rice_final.pth
    |    |--...
    |--lpr_ric_vic
    |    |--config.yaml
    |    |--rice_final.pth
    |    |--...
    |--...
```

Evaluating ICL model on LPR4M
```bash
python lpr_eval.py --config_file save/lpr_rice_vic/config.yaml
```
Evaluating ICL model on MovingFashion
```bash
python movingfashion_eval.py --config_file save/mf_rice_vic/config.yaml
```
`save/mf_rice_vic/` is the experiment path, and `config.yaml` is the training configuration.
You can specify `config_file` to evaluate other models.

### Comparing with SOTA
<table align="center">
    <thead>
        <tr>
            <th rowspan=2></th>
            <th colspan=4>LPR4M</th>
            <th colspan=4>MovingFashion</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td></td>
            <td>R1</td>
            <td>R5</td>
            <td>R10</td>
            <td>ckpt</td>
            <td>R1</td>
            <td>R5</td>
            <td>R10</td>
            <td>Checkpoints</td>
        </tr>
        <tr>
          <td>FashionNet</td>
          <td></td>
          <td></td>
          <td></td>
          <td><a href="">URL</a></td>
          <td></td>
          <td></td>
          <td></td>
          <td><a href="">URL</a></td>
        </tr>
        <tr>
          <td>TimeSFormer</td>
          <td></td>
          <td></td>
          <td></td>
          <td><a href="">URL</a></td>
          <td></td>
          <td></td>
          <td></td>
          <td><a href="">URL</a></td>
        </tr>
        <tr>
          <td>Swin-B</td>
          <td></td>
          <td></td>
          <td></td>
          <td><a href="">URL</a></td>
          <td></td>
          <td></td>
          <td></td>
          <td><a href="">URL</a></td>
        </tr>
        <tr>
          <td>RICE</td>
          <td></td>
          <td></td>
          <td></td>
          <td><a href="">URL</a></td>
          <td></td>
          <td></td>
          <td></td>
          <td><a href="">URL</a></td>
        </tr>
    </tbody>
</table>

## Training

Training ICL on LPR dataset
```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=xxx --master_port=29500 mmf_cli/run.py config=projects/videotoshop/configs/e2e_pretraining_vic.yaml model=rice dataset=videotoshop

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=xxx --master_port=29500 mmf_cli/run.py config=projects/videotoshop/configs/e2e_pretraining_vic.yaml model=rice dataset=videotoshop
```

Training ICL on MovingFashion dataset
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=xxx --master_port=29501 mmf_cli/run.py config=proje    cts/movingfashion/configs/e2e_pretraining_vic.yaml model=rice dataset=movingfashion
```
Note that the models on MovingFashion are trained via Single-Node with 2 RTX 3090 GPUs.

If you want to train other models, simply set `config` to the model configuration. 

## Citation

```bibtex
@InProceedings{Yang_2023_ICCV,
    author    = {Yang, Wenjie and Chen, Yiyi and Li, Yan and Cheng, Yanhua and Liu, Xudong and Chen, Quan and Li, Han},
    title     = {Cross-view Semantic Alignment for Livestreaming Product Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {13404-13413}
}
``` 
