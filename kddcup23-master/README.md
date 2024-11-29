# KDD Cup 2023

## 1 Environment

```
pytorch==1.12.1
cudatoolkit==10.1
recbole==1.1.1
```

## 2 Quick Start

### 2.1 Run baselines on single domain (Task 1)

**Convert a single dataset to atomic files**

```
python data_process/convert_recbole_format.py
```

**Run baseline SRGNN**

```
python run_recbole.py -m SRGNN
```

### 2.2 Transfer learning (Task 2)

**2.2.1 Data Pre-processing**

```bash
# Generate atomic files
python data_process/convert_recbole_format.py -i "sample 3/task1/uk" -o dataset/pretrain/task1_uk
python data_process/convert_recbole_format.py -i "sample 3/task1/de" -o dataset/pretrain/task1_de
python data_process/convert_recbole_format.py -i "sample 3/task1/jp" -o dataset/pretrain/task1_jp
python data_process/convert_recbole_format.py -i "sample 3/task2/fr" -o dataset/downstream/task2_fr

# Generate text embedddings
python data_process/process_text.py -i "sample 3/task1/uk" -o dataset/pretrain -d task1_uk
python data_process/process_text.py -i "sample 3/task1/de" -o dataset/pretrain -d task1_de
python data_process/process_text.py -i "sample 3/task1/jp" -o dataset/pretrain -d task1_jp
python data_process/process_text.py -i "sample 3/task2/fr" -o dataset/downstream -d task2_fr

# Merge pre-training datasets
python data_process/merge_pretrain_data.py
```

**2.2.2 Pre-train from scratch**

Pre-train on one single GPU.

```
python unisrec/pretrain.py
```

Pre-train with distributed data parallel on GPU:0-3.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python unisrec/ddp_pretrain.py
```

The pre-trained model is saved as `saved/UniSRec-dju-50.pth`.

**2.2.3 Train and evaluate on downstream datasets**

Fine-tune the pre-trained UniSRec model in transductive setting.

```
python unisrec/finetune.py -d task2_fr -p saved/UniSRec-dju-50.pth
```

Fine-tune the pre-trained model in inductive setting.

```
python unisrec/finetune.py -d task2_fr -p saved/UniSRec-dju-50.pth --train_stage=inductive_ft
```

Train UniSRec from scratch (w/o pre-training).

```
python unisrec/finetune.py -d task2_fr
```
