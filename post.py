import jsonlines
import os
import pandas as pd
import numpy as np

from statistics import mean
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from matplotlib import figure
import seaborn as sb
#pip install numba==0.57.0
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict
from ruamel.yaml import YAML


yaml = YAML(typ="safe")
params = yaml.load(open("params.yaml", encoding="utf-8"))

base_path = "../PycharmProjects/scruf_tors_2023/Movies/data/"

base_names = [
]

history_files = []
mechanisms = []

for base_name in base_names:
  history_file_path = os.path.join(base_path, f"{base_name}.json")
  mechanism_name = f"{base_name}"
  history_files.append(history_file_path)
  mechanisms.append(mechanism_name)

recs_file =
items_file =


### STATISTICS FUNCTIONS ###

# implicit type detection for pandas lookups
def typecast(series, var):
    dtype = pd.api.types.infer_dtype(series)
    dtypes = {"string":str,"integer":int,"floating":float,"mixed-integer-float":float}
    if type(var) != dtypes[dtype]:
        var = dtypes[dtype](var)
    if dtype not in dtypes.keys():
        warnings.warn("Type of column "+series.name+" could not be implicitly determined. Defaulting to integer...")
        var = int(var)
    return var

# given an item id return a list of its features as binary values
def get_item_features(item_features, item_id):
    feature_values = []
    for value in item_features.loc[(item_features.Item == typecast(item_features.Item, item_id))]["BV"]:
        feature_values.append(value)
    return feature_values


### VISUALIZATION FUNCTIONS ###

def process_history(history, fair=True, compat=True, alloc=True, lists=True):
    if fair:
        fair_list = [entry['allocation']['fairness scores'] for entry in history]
        fair_df = pd.DataFrame(fair_list)
    else:
        fair_df = None
    if compat:
        compat_list = [entry['allocation']['compatibility scores'] for entry in history]
        compat_df = pd.DataFrame(compat_list)
    else:
        compat_df = None
    if alloc:
        alloc_list = [entry['allocation']['output'] for entry in history]
        alloc_df = pd.DataFrame(alloc_list)
        alloc_df['none'] = (alloc_df['COUNTRY_low_pfr'] == 0) & (alloc_df['loan_buck_5'] == 0)
    else:
        alloc_df = None
    if lists:
        results_list = [process_results(entry['choice_out']['results']) for entry in history]
    else:
        results_list = None
    return fair_df, compat_df, alloc_df, results_list

def process_results(result_structs):
    return [(entry['item'], entry['score']) for entry in result_structs]

def plot_fairness_time(experiment_data, include_none=False, image_prefix=None):

    fair_df = experiment_data[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.set(font_scale=2)
    plt.xlabel("Time")
    plt.ylabel("Fairness")
    sb.lineplot(data=fair_df)
    image_file = image_prefix + '-fairness.png'
    plt.savefig(image_file)

def plot_allocation(experiment_data, include_none=False, image_prefix=None):
    alloc_df = pd.DataFrame(experiment_data[2])
    if include_none is False:
        if not alloc_df['none'][1:].any():
            alloc_df.drop('none', axis=1, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.set(font_scale=2)
    plt.xlabel("Time")
    plt.ylabel("Allocation")
    sb.lineplot(data=alloc_df.cumsum())
    image_file = image_prefix + '-allocation.png'
    plt.savefig(image_file)

def plot_fairness_regret(experiment_data, include_none=False, image_prefix=None):

    fair_df = experiment_data[0]
    regret = 1-fair_df
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.set(font_scale=2)
    plt.xlabel("Time")
    plt.ylabel("Fairness Regret")
    sb.lineplot(data=regret.cumsum())
    image_file = image_prefix + '-regret.png'
    plt.savefig(image_file)

def do_plots(experiment_data, include_none=False, image_prefix=None):
    plot_fairness_time(experiment_data, include_none, image_prefix)
    plot_allocation(experiment_data, include_none, image_prefix)
    plot_fairness_regret(experiment_data, include_none, image_prefix)

def process(experiment, include_none=False, image_prefix=None):
    experiment_data = process_history(experiment)
    do_plots(experiment_data, include_none, image_prefix)

def process_names(name):
    orig_name = name
    for alloc in ["Baseline","Lottery","Weighted Product","Least Fair"]:
        name = name.replace(alloc, "")
        if name != orig_name:
            return name.rstrip(), alloc

# generate array for denom of ndcg calcs
rec_list = 10
base_logs = np.log2(np.arange(rec_list)+2)

@njit
def calculate_ndcg(histories: types.DictType(types.unicode_type, types.DictType(types.unicode_type, types.float64[:])),
                  recommender: types.DictType(types.unicode_type, types.UniTuple(types.float64[:], 2)),
                  base_logs: types.float64[:]) -> types.float64[:]:
    avg_of_ndcg = []
    for history in histories:
        i_count = 0
        all_ndcg = 0
        for user, items in histories[history].items():
            scores = np.empty(0)
            for item in items:
                idx_array = np.asarray(recommender[user][1] == item).nonzero()[0]
                if idx_array.size != 0:
                    idx = idx_array[0]
                    score = recommender[user][0][idx]
                else:
                    score = 0.0
                scores = np.append(scores, score)
            ideal_scores = np.sort(recommender[user][0])[::-1][:len(scores)]
            if len(ideal_scores) < len(scores):
                ideal_scores= np.append(ideal_scores, np.zeros(len(scores) - len(ideal_scores)))
            #while len(ideal_scores) < len(scores):
            #    ideal_scores = np.append(ideal_scores, 0.0)
            scores[scores > 0] = 1.0
            ideal_scores[ideal_scores > 0] = 1.0
            recdcg = np.sum(scores/base_logs)
            idealdcg = np.sum(ideal_scores/base_logs)
            if idealdcg == 0.0:
                ndcg = 0.0
            else:
                ndcg = recdcg/idealdcg
            i_count += 1
            all_ndcg += ndcg

        avg_of_ndcg.append(all_ndcg/i_count)
    return avg_of_ndcg
agents = num_agents

for agent in agents ():


