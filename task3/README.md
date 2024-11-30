# Task 3


## Quick Start

For task 3, the test set includes products that do not appear in the training set, and participants are asked to generate the title of the next product based on the user session.
A simple T5 baseline is provided and can be executed as follows.

### Convert format

```bash
python convert_summarization_format.py
```

### Execute T5 baseline

```bash
python run_summarization.py     --model_name_or_path t5-small     --do_train     --do_eval   --source_prefix "summarize: "     --output_dir /efs/users/wjin/tmp/tst-summarization     --per_device_train_batch_size=4     --per_device_eval_batch_size=4     --overwrite_output_dir     --predict_with_generat  --train_file task3_train.csv --validation_file task3_test.csv --text_column text --summary_column summary --learning_rate 5e-3 --num_train_epochs 1
```