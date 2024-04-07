# crossview-semantic-alignment

The official code of ICCV 2023 paper
[Cross-view Semantic Alignment for Livestreaming Product Recognition](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Cross-view_Semantic_Alignment_for_Livestreaming_Product_Recognition_ICCV_2023_paper.html)
<p align="center">
  <img width="800" height="500" src="./images/model.png">
</p>

## 1.Requirements

## 2.Preparing data

## 3.Evaluation

| |    | CUHK-SYSU  |  | PRW |
| ---- |  :----:  | :----:  | :----:  | :----:  |
| Method |  mAP   | rank1  | mAP   | rank1  |
| OIM [1] | [88.1](https://drive.google.com/file/d/1Im4o0d7hytno-aycSDPHgNkxnJN785v0/view?usp=sharing)  | 89.2 | [36.0](https://drive.google.com/file/d/1l7eKIwOYJxEopguMk_tl8bKQW0f83PJv/view?usp=sharing) | 76.7 |
| NAE [4] | [89.8](https://drive.google.com/file/d/1mCCEnvwQC8Ckn7ElIJFMGqvfZMD6MX1P/view?usp=sharing)  | 90.7 | [37.9](https://drive.google.com/file/d/1zUGlmIoScRR_qFhDGdl8jtmR9cy3YrBF/view?usp=sharing) | 77.3 |
| baseline | [90.0](https://drive.google.com/file/d/17ViFt0rFNXupSNri1DvEhSFpebtqa4Xl/view?usp=sharing) | 91.0 | [40.5](https://drive.google.com/file/d/1H3f2C5GplCxxsxtKgdtdzRsX9mincwC8/view?usp=sharing) | 81.3 |

<table>
    <thead>
        <tr>
            <th>Layer 1</th>
            <th>Layer 2</th>
            <th>Layer 3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>L1 Name</td>
            <td rowspan=2>L2 Name A</td>
            <td>L3 Name A</td>
        </tr>
        <tr>
            <td>L3 Name B</td>
        </tr>
        <tr>
            <td rowspan=2>L2 Name B</td>
            <td>L3 Name C</td>
        </tr>
        <tr>
            <td>L3 Name D</td>
        </tr>
    </tbody>
</table>


<table>
    <thead>
        <tr>
            <th></th>
            <th></th>
            <th></th>
            <th colspan=4>LPR4m</th>
            <th colspan=4>MovingFashion</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ICL</td>
            <td>PMD</td>
            <td>PFR</td>
            <td>R1</td>
            <td>R5</td>
            <td>R10</td>
            <td>ckpt</td>
            <td>R1</td>
            <td>R5</td>
            <td>R10</td>
            <td>ckpt</td>
        </tr>
    </tbody>
</table>

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
