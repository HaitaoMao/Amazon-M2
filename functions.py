import os
import numpy as np
import pandas as pd
import ast
from collections import Counter, defaultdict, OrderedDict
import re
import math
import numbers
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
from nltk.corpus import words
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# Define stop words
from functools import lru_cache




@lru_cache(maxsize=1)
def read_product_data(train_data_dir='.'):
    return pd.read_csv(os.path.join(train_data_dir, 'products_train.csv'))

@lru_cache(maxsize=1)
def read_train_data(train_data_dir='.'):
    sessions = pd.read_csv(os.path.join(train_data_dir, 'sessions_train.csv'))
    sessions.prev_items = sessions.prev_items.apply(split)
    return sessions

@lru_cache(maxsize=3)
def read_test_data(task=1, test_data_dir='.'):
    sessions = pd.read_csv(os.path.join(test_data_dir, f'sessions_test_{task}.csv'))
    sessions.prev_items = sessions.prev_items.apply(split)
    return sessions



def read_locale_data(locale, task):
    products = read_product_data().query(f'locale == "{locale}"')
    sess_train = read_train_data().query(f'locale == "{locale}"')
    sess_test = read_test_data(task).query(f'locale == "{locale}"')
    return products, sess_train, sess_test


def show_locale_info(locale, task):
    
    products, sess_train, sess_test = read_locale_data(locale, task)
    sess_test1 = read_test_data("task1")
    sess_test2 = read_test_data("task2")
    sess_test = pd.concat([sess_test1, sess_test2])
    
    train_l = sess_train['prev_items'].apply(lambda sess: len(sess))
    
    test_l = sess_test['prev_items'].apply(lambda sess: len(sess))
    if len(sess_test) > 0:
        print(
            f"Locale: {locale} \n"
            f"Test session lengths -"
            f"Number of test sessions: {len(sess_test)} \n"
            f"Mean: {test_l.mean():.2f}  "
            f"Number of interaction: {test_l.sum():.2f}"
        )
    print("======================================================================== \n")


def change_format(datas):
    new_datas = {}
    datas = datas.tolist()
    for idx, data in enumerate(datas):
        new_datas[str(idx)] = {'category': data[0], 'same_cate': data[1:]}
    return new_datas


@lru_cache(maxsize=1)
def read_product_data(train_data_dir="."):
    datas = pd.read_csv(os.path.join(train_data_dir, 'products_train.csv'))
    return datas

def load_whole():
    with open("filtered_data/all_sessions.txt", "rb") as f:
        all_sessions = pickle.load(f)
    with open("filtered_data/sessions.txt", "rb") as f:
        sessions = pickle.load(f)

    return sessions, all_sessions

def cal_frequency(sessions):
    items = sessions.prev_items.explode()
    counts = Counter(items)
    counts.update(sessions.next_item.explode())
    counts = list(dict(counts).values())
    
    counts = np.array(sorted(counts))
    num_all_query = len(counts)
    num_queries, frequencies = [0], [num_all_query]
    unique_frequencies = np.unique(counts).tolist()
    for unique_frequency in unique_frequencies:
        num_all_query -= np.sum(counts == unique_frequency)
        num_queries.append(unique_frequency)
        frequencies.append(num_all_query)
    
    return num_queries, frequencies


def cal_length(sessions):
    lengths = sessions.prev_items.apply(lambda x: len(x))
    counts = Counter(lengths)
    counts = list(dict(counts).values())

    return lengths, counts

def cal_overlap(session_dict):
    locale_keys = list(session_dict.keys())
    num_locales = len(locale_keys)
    data = np.zeros([num_locales, num_locales])
    for i, locale1 in enumerate(locale_keys):
        for j, locale2 in enumerate(locale_keys):
            if i == j: 
                data[i][j] = 1
                continue
            if i > j: continue
            sessions1, sessions2 = session_dict[locale1].prev_items, session_dict[locale2].prev_items
            item1 = set(item for session in sessions1 for item in session)            
            item2 = set(item for session in sessions2 for item in session)
            
            num_item1, num_item2 = len(item1), len(item2)
            num_item = len(item1.intersection(item2))
            data[i][j] = num_item / num_item1
            data[j][i] = num_item / num_item2


    data_dict = {}
    for i, key in enumerate(locale_keys):
        data_dict[key] = data[i]
    df = pd.DataFrame(data_dict)
    df = df.round(2)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set(font_scale=2)
    ax = sns.heatmap(df, vmin=0, vmax=1, annot=True, yticklabels=locale_keys, ax=ax) # , yticklabels=y_ticks
    ax.tick_params(axis='x', length=0, pad=10)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), fontsize=32, fontfamily='serif') # , rotation=45, ha='right'
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=32, fontfamily='serif')
    
    plt.savefig(f"fig/overlap2/overlap.png", bbox_inches='tight' )
    plt.savefig(f"fig/overlap2/overlap.pdf", bbox_inches='tight')

    return data



def cal_repeat(sessions):
    lengths = sessions.prev_items.apply(lambda x: len(x))
    unique_lengths = sessions.prev_items.apply(lambda x: len(set(x)))
    num_repeat = np.sum(lengths != unique_lengths)
    num_unrepeat = np.sum(lengths == unique_lengths)
    repeat_ratio = (lengths - unique_lengths) / lengths
    repeat_mask = (lengths != unique_lengths)
    repeat_lengths = lengths[repeat_mask] - unique_lengths[repeat_mask] 
    lengths, unique_lengths, repeat_mask, repeat_lengths = lengths.tolist(), unique_lengths.tolist(), repeat_mask.tolist(), repeat_lengths.tolist()
    prev_items = sessions.prev_items.tolist()
    

    repeat_idx = 0
    if np.sum(repeat_mask) > 0:
        for idx in range(len(repeat_mask)):
            if repeat_mask[idx]:
                counts = Counter(prev_items[idx])
                counts = list(dict(counts).values())
                counts = np.array(counts)
                above_counts = np.sum(np.clip(counts - 2, a_min=0, a_max=None))
                repeat_lengths[repeat_idx] -= above_counts
                repeat_idx += 1
    else:
        num_repeat = 0
        repeat_lengths = [0]

    return num_repeat / len(lengths), repeat_lengths
    
def generate_sessions_with_locales(origin_sessions, is_merge=True):
    locale_names = origin_sessions['locale'].unique()
    sessions = {}

    for locale_name in locale_names:
        locale_sessions = origin_sessions.query(f'locale == "{locale_name}"')
        locale_sessions.prev_items = locale_sessions.prev_items.apply(split)

        if is_merge:
            locale_sessions.prev_items = [prev_items + [next_item] for prev_items, next_item in zip(locale_sessions.prev_items, locale_sessions.next_item)]
        sessions[locale_name] = locale_sessions
    
    all_sessions = pd.concat(list(sessions.values()), ignore_index=True)

    return sessions, all_sessions

def split(string):
    words = string.strip("[ ]").split()
    words = [word.replace("\'", '') for word in words]

    return words


