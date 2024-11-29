import pandas as pd
from tqdm import tqdm

# train_df = pd.read_csv('./amazon/amazon.train.inter', sep='\t')
# test_df = pd.read_csv('./amazon/amazon.test.inter', sep='\t')

# train_popular = train_df[train_df['item_locale:token'].isin(['UK', 'JP', 'DE'])]
# train_unpopular = train_df[train_df['item_locale:token'].isin(['ES', 'FR', 'IT'])]
# test_popular = test_df[test_df['item_locale:token'].isin(['UK', 'JP', 'DE'])]
# test_unpopular = test_df[test_df['item_locale:token'].isin(['ES', 'FR', 'IT'])]
# train_popular.to_csv('./amazon_popular/amazon_popular.train.inter', sep='\t', index=False)
# train_unpopular.to_csv('./amazon_unpopular/amazon_unpopular.train.inter', sep='\t', index=False)
# test_popular.to_csv('./amazon_popular/amazon_popular.test.inter', sep='\t', index=False)
# test_unpopular.to_csv('./amazon_unpopular/amazon_unpopular.test.inter', sep='\t', index=False)

# train_ES = train_df[train_df['item_locale:token'].isin(['ES'])]
# train_FR = train_df[train_df['item_locale:token'].isin(['FR'])]
# train_IT = train_df[train_df['item_locale:token'].isin(['IT'])]
# test_ES = test_df[test_df['item_locale:token'].isin(['ES'])]
# test_FR = test_df[test_df['item_locale:token'].isin(['FR'])]
# test_IT = test_df[test_df['item_locale:token'].isin(['IT'])]
# train_ES.to_csv('./amazon_ES/amazon_ES.train.inter', sep='\t', index=False)
# train_FR.to_csv('./amazon_FR/amazon_FR.train.inter', sep='\t', index=False)
# train_IT.to_csv('./amazon_IT/amazon_IT.train.inter', sep='\t', index=False)
# test_ES.to_csv('./amazon_ES/amazon_ES.test.inter', sep='\t', index=False)
# test_FR.to_csv('./amazon_FR/amazon_FR.test.inter', sep='\t', index=False)
# test_IT.to_csv('./amazon_IT/amazon_IT.test.inter', sep='\t', index=False)

def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split(' ') if i]
    return l

df_train = pd.read_csv('./amazon/amazon.train.inter', sep='\t')
df_test = pd.read_csv('./amazon/amazon.test.inter', sep='\t')
all_items = set()
for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
    prev_items = str2list(row['item_id_list:token_seq'])
    next_item = row['item_id:token']
    all_items.add(next_item)
    all_items.update(prev_items)

for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    prev_items = str2list(row['item_id_list:token_seq'])
    next_item = row['item_id:token']
    all_items.add(next_item)
    all_items.update(prev_items)


test_df1 = pd.read_csv('./task2phase2/sessions_test_task2.csv', sep=',')
test_df1['item_id_list:token_seq'] = test_df1['prev_items'].apply(lambda x: ' '.join([str(i) for i in str2list(x) if i in all_items])) 
test_df1['item_locale:token'] = test_df1['locale']
test_df1['item_id:token'] = None
test_df1['session_id:token'] = [100000000+i for i in range(len(test_df1))]
test_df1 = test_df1.drop(columns=['prev_items', 'locale'])
# import ipdb; ipdb.set_trace()
test_df1['item_id_list:token_seq'] = test_df1['item_id_list:token_seq'].apply(lambda x: 'B08GYKNCCP' if x == '' else x)
df_test.append(test_df1, ignore_index=True).to_csv('./task2phase2/task2phase1.test.inter', sep='\t', index=False)
# import ipdb; ipdb.set_trace()
# session_id:token        item_id_list:token_seq  item_id:token   item_locale:token

