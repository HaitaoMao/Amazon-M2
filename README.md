# Amazon M2

This is the official implementation for [Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation](https://arxiv.org/pdf/2307.09688.pdf) with baselines. The instructions of the KDD CUP challenge can be found at [here](https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge)

## Datasets

### Amazon Datasets

The folders with the prefix `amazon` contain both the `train` and `test` sets. 

### Released Datasets

For all of the three tasks `task1`, `task2` and `task3`, the private test sets of the KDD CUP [Challenges]((https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge)) are also included under the folder `dataset`

## Baselines

We provide our implementation on the following representative baselines.

1. [CORE](https://github.com/RUCAIBox/CORE)

2. [GRU4Rec](https://github.com/hidasib/GRU4Rec)

3. [NARM](https://github.com/lijingsdu/sessionRec_NARM)

4. [SRGNN](https://github.com/CRIPAC-DIG/SR-GNN)

5. [STAMP](https://github.com/uestcnlp/STAMP)


## Example Usages


To train the model
```
usage: python train.py [--dataset DATASET] [--gpu GPU_ID] [--model MODEL] [--lr LR] [--dropout=DROPOUT_RATIO] [--num_layers=NUM_LAYERS]
```

To further fine-tune the model
```
python finetune.py [--dataset DATASET] [--gpu GPU_ID] [--model MODEL] [--lr LR] [--dropout=DROPOUT_RATIO] [--num_layers=NUM_LAYERS][--saved_model=PRETRAINED_MODEL_PATH]
```

We provide some example usage under the folder `baseline` with the shell scripts.

## Requirements

```
conda install -c aibox recbole=1.2.0
```


## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

Please cite the following papers as the references if you use our codes or the processed datasets.

```
@article{jin2024amazon,
  title={Amazon-m2: A multilingual multi-locale shopping session dataset for recommendation and text generation},
  author={Jin, Wei and Mao, Haitao and Li, Zheng and Jiang, Haoming and Luo, Chen and Wen, Hongzhi and Han, Haoyu and Lu, Hanqing and Wang, Zhengyang and Li, Ruirui and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}


@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={{CIKM}},
  year={2021}
}
```