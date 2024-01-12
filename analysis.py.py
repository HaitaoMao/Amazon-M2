import os
import numpy as np
import pandas as pd
from functions import *
from plot import *
import pickle
from KNN import SKNN
from collections import Counter

train_data_dir = '.'
test_data_dir = '.'
task = 'task1'
PREDS_PER_SESSION = 100


# Cache loading of data for multiple calls

products = read_product_data()

locale_names = products['locale'].unique()
test1_data = read_test_data("task1")
test2_data = read_test_data("task2")
for locale_name in locale_names:
    show_locale_info(locale_name, "task1")
    show_locale_info(locale_name, "task2")
# ['DE', 'JP', 'UK', 'ES', 'FR', 'IT']

sessions, all_sessions = load_whole()
xs, ys = cal_frequency(all_sessions)
plot_frequency(xs, ys, "all")

# overlapping between different regions. 
cal_overlap(sessions)

# session length distribution for all sessions
cal_length(all_sessions)

# session length distribution per locale
keys = ["UK"]
for key in sessions.keys():
    sessions_locale = sessions[key]
    lengths, counts = cal_length(sessions_locale)
    dist_plot_single(lengths, key)

repeat_ratios = {}
keys = list(sessions.keys())

# repeat pattern for all sessions
repeat_ratio, repeat_lengths = cal_repeat(all_sessions)
repeat_ratios["all"] = repeat_ratio
dist_plot_repeat(repeat_lengths, "all")

# repeat patterns for each locale
keys = ["JP"]
for key in keys:
    sessions_locale = sessions[key]
    repeat_ratio, repeat_lengths = cal_repeat(sessions_locale)
    repeat_ratios[key] = repeat_ratio
    dist_plot_repeat(repeat_lengths, key)


# retrivial KNN
new_sessions = {}
all_sessions = np.array([session for session in all_sessions])
for key in sessions.keys():
    sessions_locale = sessions[key]
    sessions_locale = [session_locale for session_locale in sessions_locale]
    new_sessions[key] = np.array(sessions_locale)
sessions = new_sessions

session_ids = np.arange(len(all_sessions))
model = SKNN(session_ids, all_sessions, sample_size=10, k=10)

results = []
for session_id in range(len(all_sessions)):
    result = model.predict(session_id, all_sessions[session_id], k=10)
    result = [items[1] for items in result]
    results.append(result)

new_results = []
for result in results:
    new_results.append(np.median(result))
    
dist_plot_coll(new_results, "all")




