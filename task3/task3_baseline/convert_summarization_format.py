import os
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import swifter
import nltk
import evaluate

base = 'KDDCUP23/'
products = pd.read_parquet(os.path.join(base, 'task1', 'products.parquet'))
sessions_train = pd.read_parquet(os.path.join(base, 'task1', 'sessions_train.parquet'))

only_locale = 'UK'
if only_locale is not None:
    products = products[products.locale==only_locale]
    sessions_train = sessions_train[sessions_train.locale==only_locale]

id2title = {id: title for id, title in products[['id', 'title']].values}
titles = sessions_train.prev_items.explode().apply(lambda x: id2title[x])
titles_prev = titles.groupby(level=0).apply(lambda x: ' '.join(x))
titles_next = sessions_train.next_item.apply(lambda x: id2title[x])
data = pd.DataFrame({'text': titles_prev, 'summary': titles_next})
train_data.to_csv('task3_train.csv', index=False)

test_prev = pd.read_parquet(os.path.join(base, 'task3', 'public_test', 'sessions_test.parquet'))
gt_product2 = pd.read_parquet(os.path.join(base, 'task3', 'public_test', 'DONT_SHARE_gt_products.parquet'))
if only_locale is not None:
    test_prev = test_prev[test_prev.locale == only_locale]
    gt_product2 = gt_product2[gt_product2.locale == only_locale]

titles = test_prev.prev_items.explode().apply(lambda x: id2title[x])
titles_prev = titles.groupby(level=0).apply(lambda x: ' '.join(x))
titles_next = gt_product2.title
test_data = pd.DataFrame({'text': titles_prev, 'summary': titles_next})
test_data.to_csv('task3_test.csv', index=False)

