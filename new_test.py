import pandas as pd
import ast

method = 'CORE'
# method = 'popularity'
# dataset = 'amazon_popular'
# dataset = 'amazon_unpopular'
dataset = 'amazon'
df = pd.read_csv(f'{method}_{dataset}.csv', sep='\t')
# df = pd.read_csv(f'{dataset}/{method}_{dataset}.csv', sep='\t')
# df = pd.read_csv(f'{dataset}/{method}_{dataset}_all.csv', sep='\t')
# df_truth = pd.read_parquet('./task2p2.parquet')
# df['item_id:token'] = df_truth['next_item']
df['next_item_prediction'] = df['next_item_prediction'].apply(ast.literal_eval)


import pandas as pd
import numpy as np

def dcg_at_k(r, k):
    """Compute DCG for a given list of relevances up to position k."""
    r = np.asfarray(r)[:k]
    return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2)))

def ndcg_at_k(predicted_list, true_item, k):
    """Compute NDCG for predictions up to position k."""
    # Create a binary list where 1 indicates a hit, and 0 indicates a miss
    r = [1 if item == true_item else 0 for item in predicted_list]

    # Compute DCG for the predicted list
    dcg_max = dcg_at_k(r, k)

    # Compute IDCG for the perfect list
    idcg = dcg_at_k(sorted(r, reverse=True), k)

    # Handle edge case where IDCG is 0
    if not idcg:
        return 0.
    return dcg_max / idcg


def get_rank(row):
    try:
        return row['next_item_prediction'].index(row['item_id:token']) + 1
    except ValueError:
        return 0

def get_recall(row):
    return row['item_id:token'] in row['next_item_prediction']

def caculate_mrr(df_):
    df_['rank'] = df_.apply(get_rank, axis=1)
    mrr = df_['rank'].apply(lambda x: 1/x if x>0 else 0).mean()
    return mrr

def caculate_recall(df_):
    df_['recall'] = df_.apply(get_recall, axis=1)
    recall_at_100 = df_['recall'].mean()
    return recall_at_100

def caculate_ndcg(df_):
    ndcgs = []
    for index, row in df_.iterrows():
        # Assuming 'next_item_prediction' is a string of items separated by commas
        predicted_list = row['next_item_prediction']
        true_item = row['item_id:token']
        ndcgs.append(ndcg_at_k(predicted_list, true_item, 100))
    return np.mean(ndcgs)

# df_UK = df[df['item_locale:token']=='UK'].copy()
# df_DE = df[df['item_locale:token']=='DE'].copy()
# df_JP = df[df['item_locale:token']=='JP'].copy()

# print('overall', 'mrr', caculate_mrr(df), 'recall', caculate_recall(df), 'ndcg', caculate_ndcg(df))
# print('UK', 'mrr', caculate_mrr(df_UK), 'recall', caculate_recall(df_UK), 'ndcg', caculate_ndcg(df_UK))
# print('DE', 'mrr', caculate_mrr(df_DE), 'recall', caculate_recall(df_DE), 'ndcg', caculate_ndcg(df_DE))
# print('JP', 'mrr', caculate_mrr(df_JP), 'recall', caculate_recall(df_JP), 'ndcg', caculate_ndcg(df_JP))

df_ES = df[df['item_locale:token']=='ES'].copy()
df_FR = df[df['item_locale:token']=='FR'].copy()
df_IT = df[df['item_locale:token']=='IT'].copy()
# import ipdb; ipdb.set_trace()
print('overall', 'mrr', caculate_mrr(df), 'recall', caculate_recall(df), 'ndcg', caculate_ndcg(df))
print('ES', 'mrr', caculate_mrr(df_ES), 'recall', caculate_recall(df_ES), 'ndcg', caculate_ndcg(df_ES))
print('FR', 'mrr', caculate_mrr(df_FR), 'recall', caculate_recall(df_FR), 'ndcg', caculate_ndcg(df_FR))
print('IT', 'mrr', caculate_mrr(df_IT), 'recall', caculate_recall(df_IT), 'ndcg', caculate_ndcg(df_IT))
