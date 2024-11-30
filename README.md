# Amazon M2

This is the official implementation for [Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation](https://arxiv.org/pdf/2307.09688.pdf) accepted by NeurIPS'22. Moreover, the dataset is utilized a competition in the KDD CUP 2023 and have attracted thousands of teams and submissions. The instructions of the KDD CUP challenge can be found at [here](https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge)

## Abstract

Modeling customer shopping intentions is a crucial task for e-commerce, as it directly impacts user experience and engagement. Thus, accurately understanding customer preferences is essential for providing personalized recommendations. Session-based recommendation, which utilizes customer session data to predict their next interaction, has become increasingly popular. However, existing session datasets have limitations in terms of item attributes, user diversity, and dataset scale. As a result, they cannot comprehensively capture the spectrum of user behaviors and preferences. To bridge this gap, we present the Amazon Multilingual Multi-locale Shopping Session Dataset, namely Amazon-M2. It is the first multilingual dataset consisting of millions of user sessions from six different locales, where the major languages of products are English, German, Japanese, French, Italian, and Spanish. Remarkably, the dataset can help us enhance personalization and understanding of user preferences, which can benefit various existing tasks as well as enable new tasks. To test the potential of the dataset, we introduce three tasks in this work: (1) next-product recommendation, (2) next-product recommendation with domain shifts, and (3) next-product title generation. With the above tasks, we benchmark a range of algorithms on our proposed dataset, drawing new insights for further research and practice. 


## Datasets


### Tasks
1. Next Product Recommendation
2. Next Product Recommendation for Underrepresented Languages/Locales
3. Next Product Title Generation

### Dataset Description

The dataset consists of two main components: user sessions and product attributes. User sessions are a list of products that a user has engaged with in chronological order, while product attributes include various details like product title, price in local currency, brand, colour, and description.

The dataset has been divided into three splits: train, phase-1 test, and phase-2 test. For Task 1 and Task 2, the proportions for each language are roughly 10:1:1. For Task 3, the number of samples in the phase-1 test and phase-2 test is fixed at 10,000. All three tasks share the same train set, while their test sets have been constructed according to their specific objectives. Task 1 uses English, German, and Japanese data, while Task 2 uses French, Italian, and Spanish data. Participants in Task 2 are encouraged to use transfer learning to improve their system's performance on the test set. For Task 3, the test set includes products that do not appear in the training set, and participants are asked to generate the title of the next product based on the user session.

### Amazon Datasets

The folders with the prefix `amazon` contain both the `train` and `test` sets. 



### Privated Released Datasets

For all of the three tasks `task1`, `task2` and `task3`, the private test sets of the KDD CUP [Challenges]((https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge)) are also included under the folder `task1`, `task2` and `task3` respectively.




## Baselines for Task 1 & 2

For task 1 & 2, we provide our implementation on the following representative baselines.

1. [CORE](https://github.com/RUCAIBox/CORE)
2. [GRU4Rec++](https://github.com/hidasib/GRU4Rec)
3. [NARM](https://github.com/lijingsdu/sessionRec_NARM)
4. [SRGNN](https://github.com/CRIPAC-DIG/SR-GNN)
5. [STAMP](https://github.com/uestcnlp/STAMP)

For task 3, we provide a simple T5 baseline

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

We have provided the yaml file of conda environment.
Once the following command is executed,
```
conda env create -f environment.yml
```
here should be a conda environment titled as `amazon`, which can be activated via `conda activate amazon`


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
