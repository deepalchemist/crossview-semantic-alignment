# crossview-semantic-alignment

The official code of ICCV 2023 paper
[Cross-view Semantic Alignment for Livestreaming Product Recognition](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Cross-view_Semantic_Alignment_for_Livestreaming_Product_Recognition_ICCV_2023_paper.html)
<p align="center">
  <img width="800" height="500" src="./images/model.png">
</p>

## 1.Requirements

## 2.Preparing data

## 3.Evaluation

We evaluate our methods on two datasets, i.e., LPR4M and MovingFashion.
<table>
    <thead>
        <tr>
            <th colspan=3></th>
            <th colspan=2>LPR4m</th>
            <th colspan=2>MovingFashion</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ICL</td>
            <td>PMD</td>
            <td>PFR</td>
            <td>R1</td>
            <td>ckpt</td>
            <td>R1</td>
            <td>ckpt</td>
        </tr>
        <tr>
          <td>&#10004</td>
          <td></td>
          <td></td>
          <td>22.63</td>
          <td><a href="https://drive.google.com/file/d/1DKJRDzsYAih_LBe2eTaeIF8hK3-J6rnU/view?usp=drive_link">URL</a></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <td>&#10004</td>
          <td>&#10004</td>
          <td></td>
          <td>25.99</td>
          <td><a href="https://drive.google.com/file/d/1X-cNDd8k-0NItx9-8CDqPclaxLQJlk6h/view?usp=drive_link">URL</a></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <td>&#10004</td>
          <td>&#10004</td>
          <td>&#10004</td>
          <td>27.17</td>
          <td><a href="https://drive.google.com/file/d/12Mu1QuVjLs2GbU2e7NcsHMzQJqWMMOmf/view?usp=drive_link">URL</a></td>
          <td></td>
          <td></td>
        </tr>
    </tbody>
</table>
The url of the trained models are available in the table. 
We trained the model using the following configuration: 2 RTX 3090 GPUs with 24GB of memory each, a global batch size of 96, and 750K sampled (video, image) trainset pairs. For the rest of the configuration, please refer to the training script.

## 4.Training

## 5.Citation

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
