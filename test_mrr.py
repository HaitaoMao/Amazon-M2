import pandas as pd
import ast

method = 'popularity'
# method = 'GRU4Rec'
dataset = 'task2phase2'
# df = pd.read_csv(f'{dataset}/{method}_{dataset}.csv', sep='\t')
df = pd.read_csv(f'{dataset}/{method}_{dataset}_all.csv', sep='\t')
df_truth = pd.read_parquet('./task2p2.parquet')
df['item_id:token'] = df_truth['next_item']
df['next_item_prediction'] = df['next_item_prediction'].apply(ast.literal_eval)

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

# df_UK = df[df['item_locale:token']=='UK'].copy()
# df_DE = df[df['item_locale:token']=='DE'].copy()
# df_JP = df[df['item_locale:token']=='JP'].copy()

# print('overall', 'mrr', caculate_mrr(df), 'recall', caculate_recall(df))
# print('UK', 'mrr', caculate_mrr(df_UK), 'recall', caculate_recall(df_UK))
# print('DE', 'mrr', caculate_mrr(df_DE), 'recall', caculate_recall(df_DE))
# print('JP', 'mrr', caculate_mrr(df_JP), 'recall', caculate_recall(df_JP))

df_ES = df[df['item_locale:token']=='ES'].copy()
df_FR = df[df['item_locale:token']=='FR'].copy()
df_IT = df[df['item_locale:token']=='IT'].copy()
print('overall', 'mrr', caculate_mrr(df), 'recall', caculate_recall(df))
print('ES', 'mrr', caculate_mrr(df_ES), 'recall', caculate_recall(df_ES))
print('FR', 'mrr', caculate_mrr(df_FR), 'recall', caculate_recall(df_FR))
print('IT', 'mrr', caculate_mrr(df_IT), 'recall', caculate_recall(df_IT))
