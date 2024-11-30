import warnings
warnings.simplefilter('ignore')

import gc
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

dataset = 'task2phase2'

df_train = pd.read_csv(f'./{dataset}/{dataset}.train.inter', sep='\t')
df_test = pd.read_csv(f'./{dataset}/{dataset}.test.inter', sep='\t')


def str2list(x):
    # x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split(' ') if i]
    return l
    
next_item_dict = defaultdict(list)

for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
    prev_items = str2list(row['item_id_list:token_seq'])
    next_item = row['item_id:token']
    prev_items_length = len(prev_items)
    if prev_items_length <= 1:
        next_item_dict[prev_items[0]].append(next_item)
    else:
        for i, item in enumerate(prev_items[:-1]):
            next_item_dict[item].append(prev_items[i+1])
        next_item_dict[prev_items[-1]].append(next_item)
        
for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    prev_items = str2list(row['item_id_list:token_seq'])
    prev_items_length = len(prev_items)
    if prev_items_length <= 1:
        continue
    else:
        for i, item in enumerate(prev_items[:-1]):
            next_item_dict[item].append(prev_items[i+1])
            
next_item_map = {}

for item in tqdm(next_item_dict):
    counter = Counter(next_item_dict[item])
    next_item_map[item] = [i[0] for i in counter.most_common(100)]
    
k = []
v = []

for item in next_item_dict:
    k.append(item)
    v.append(next_item_dict[item])
    
df_next = pd.DataFrame({'item': k, 'next_item': v})
df_next = df_next.explode('next_item').reset_index(drop=True)

top200 = df_next['next_item'].value_counts().index.tolist()[:200]

df_test['last_item'] = df_test['item_id_list:token_seq'].apply(lambda x: str2list(x)[-1])
df_test['next_item_prediction'] = df_test['last_item'].map(next_item_map)

preds = []

for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    pred_orig = row['next_item_prediction']
    pred = pred_orig
    prev_items = str2list(row['item_id_list:token_seq'])
    if type(pred) == float:
        pred = top200[:100]
    else:
        if len(pred_orig) < 100:
            for i in top200:
                if i not in pred_orig and i not in prev_items:
                    pred.append(i)
                if len(pred) >= 100:
                    break
        else:
            pred = pred[:100]
    preds.append(pred)
    
df_test['next_item_prediction'] = preds
# df_test = df_test.drop(df_test.index[:327049])
df_test = df_test.drop(df_test.index[:360625])
# df_test['next_item_prediction'] = predictions[33576:]
df_test.to_csv(f'{dataset}/popularity_{dataset}_all.csv', sep='\t', index=False)
# import ipdb; ipdb.set_trace()

# df_test[['locale', 'next_item_prediction']].to_parquet('submission_task1.parquet', engine='pyarrow')


