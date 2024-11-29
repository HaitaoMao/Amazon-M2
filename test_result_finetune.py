import os
import re

dataset = 'amazon'
model = 'GRU4Rec'
path = './log/' + model

best_recall = 0
best_recall_mrr = 0
best_mrr = 0
best_mrr_recall = 0
best_recall_file = None
best_mrr_file = None
for file in os.listdir(path):
    if dataset not in file:
        continue
    file = os.path.join(path, file)
    with open(file, 'r') as f:
        lines = f.readlines()
        if 'Finetune' not in lines[-1]:
            continue
        # import ipdb; ipdb.set_trace()
        if 'test result' in lines[-2]:
            recall_pattern = r"('recall@100', (\d+\.\d+))"
            mrr_pattern = r"('mrr@100', (\d+\.\d+))"
            recall_match = re.search(recall_pattern, lines[-2])
            mrr_match = re.search(mrr_pattern, lines[-2])
            recall_value = float(recall_match.group(2)) if recall_match else None
            mrr_value = float(mrr_match.group(2)) if mrr_match else None
            if recall_value > best_recall:
                best_recall = recall_value
                best_recall_mrr = mrr_value
                best_recall_file = file
            if mrr_value > best_mrr:
                best_mrr = mrr_value
                best_mrr_file = file
                best_mrr_recall = recall_value
print('best recall: ', best_recall, 'mrr:', best_recall_mrr, 'file:', best_recall_file)
print('best mrr: ', best_mrr, 'recall:', best_mrr_recall, 'file:', best_mrr_file)
            
            
