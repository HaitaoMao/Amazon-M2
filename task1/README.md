# Task 1

## Quick Start

For task 1, we provide our implementation on the following representative baselines.

1. [CORE](https://github.com/RUCAIBox/CORE)
2. [GRU4Rec++](https://github.com/hidasib/GRU4Rec)
3. [NARM](https://github.com/lijingsdu/sessionRec_NARM)
4. [SRGNN](https://github.com/CRIPAC-DIG/SR-GNN)
5. [STAMP](https://github.com/uestcnlp/STAMP)

### Run baselines on single domain

**Convert a single dataset to atomic files**

```
python data_process/convert_recbole_format.py
```

**Run baseline SRGNN**

```
python run_recbole.py [-m MODEL]

Example Usage: python run_recbole.py -m SRGNN
```
