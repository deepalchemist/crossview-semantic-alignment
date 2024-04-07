# crossview-semantic-alignment

The official code of ICCV 2023 paper
[Cross-view Semantic Alignment for Livestreaming Product Recognition](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Cross-view_Semantic_Alignment_for_Livestreaming_Product_Recognition_ICCV_2023_paper.html)

In the original paper, we reported the performance of the model trained on the complete dataset of 4 million samples using 8 V100 GPUs. However, we had concerns about the accessibility of the dataset and the reproducibility of the model performances. The reasons are as follows. On one hand, the large size of 4 million samples makes the data difficult to download (time-consuming and error-prone) and requires large storage space (approximately 4 TB). On the other hand, Using 8 V100 GPUs sets high resource requirements for model training. Considering these factors, we made the following adjustments in this repository: firstly, we reduced the number of training samples. We sampled a training subset containing ***750K*** samples from the full 4 million samples, which only reduced the number of samples of the head categories without sampling the tail categories. Secondly, we used the more widely available ***RTX 3090 GPU*** instead of ***V100 GPU*** to train the model, facilitating the reproduction of model results.

<p align="center">
  <img width="600" height="400" src="./images/model.png">
</p>

## 1. Requirements

## 2. Preparing data

## 3. Evaluation

We perform ablation study on LPR4M.
<table class="center">
    <thead>
        <tr>
            <th colspan=3></th>
            <th colspan=2>LPR4M</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ICL</td>
            <td>PMD</td>
            <td>PFR</td>
            <td>R1</td>
            <td>ckpt</td>
        </tr>
        <tr>
          <td>&#10004</td>
          <td></td>
          <td></td>
          <td>22.63</td>
          <td><a href="https://drive.google.com/file/d/1DKJRDzsYAih_LBe2eTaeIF8hK3-J6rnU/view?usp=drive_link">URL</a></td>
        </tr>
        <tr>
          <td>&#10004</td>
          <td>&#10004</td>
          <td></td>
          <td>25.99</td>
          <td><a href="https://drive.google.com/file/d/1X-cNDd8k-0NItx9-8CDqPclaxLQJlk6h/view?usp=drive_link">URL</a></td>
        </tr>
        <tr>
          <td>&#10004</td>
          <td>&#10004</td>
          <td>&#10004</td>
          <td>27.17</td>
          <td><a href="https://drive.google.com/file/d/12Mu1QuVjLs2GbU2e7NcsHMzQJqWMMOmf/view?usp=drive_link">URL</a></td>
        </tr>
    </tbody>
</table>

The url of the trained models are available in the table. 
We train the model using the following configuration: 2 RTX 3090 GPUs with 24GB of memory each and a global batch size of 96. For the rest of the configuration, please refer to the training script in  `./scripts`. Note that, On LPR4M, this project uses 750K training samples and a batch size of 96, while the original paper used 4 million training samples and a batch size of 256, thus the performance reported in this project is lower than that in the paper. However, as a baseline model for the proposed dataset, the performance level is not a key factor and does not affect readers from following this work.

 

Evaluating ICL on LPR4M
```bash
python lpr4m_embedding_eval.py --data_root /lpr4m/data/root/ --n_gpu 2 --sim_header mean_pooling  --one_stage --embedding_sim --ckpt_path /checkpoint/path
```

Evaluating ICL+PMD on LPR4M
```bash
python lpr4m_embedding_eval.py --data_root /lpr4m/data/root/ --n_gpu 2 --sim_header cross_attention --cross_num_hidden_layers 2 --embedding_sim --ckpt_path /checkpoint/path
```

Evaluating ICL+PMD+PFR on LPR4M
```bash
python lpr4m_embedding_eval.py --data_root /lpr4m/data/root/ --n_gpu 2 --sim_header cross_attention --cross_num_hidden_layers 2 --recons_feat --embedding_sim --ckpt_path /checkpoint/path
```
The evaluation script for each model on MovingFashion is similar to that for LPR4M.

## 4. Training
## 5. Citation

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
