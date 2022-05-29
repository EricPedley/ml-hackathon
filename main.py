import os
from enum import Enum, auto
import pandas as pd
import pandas.core.series#for type hint that helps intellisense
import numpy as np
from autolearn import run_regression_ensemble, run_classification_ensemble
from column_analysis import load_clean


data_folder_names = os.listdir('data')

class TaskTypes(Enum):#very cool python feature, thanks Thomas
    REGRESSION = auto()
    BINARY_CLASSIFICATION = auto()
    MULTI_CLASSIFICATION = auto()


def get_task_type(y_col: pandas.core.series.Series):
    #TODO: See if the metric we're using to identify regression problems is correct.
    # Right now it's assuming regression if more than 90% of the values are unique, which
    # seems right but maybe some multi class datasets have that many classes.
    if pandas.api.types.is_float_dtype(y_col) or len(pd.unique(y_col)) > 0.9 * len(y_col):
        return TaskTypes.REGRESSION
    if len(pd.unique(y_col)) == 2:
        return TaskTypes.BINARY_CLASSIFICATION
    return TaskTypes.MULTI_CLASSIFICATION

def process_uci_dataset(folder):
    folder_contents = os.listdir(f'data/{folder}')
    return 'NOT IMPLEMENTED'

def process_kaggle_dataset(folder):
    data = load_clean(f'data/{folder}/train.csv')
    target_col = ''
    for candidate_col_name in ('y', 'Y', 'target', 'Target', 'TARGET'):
        if candidate_col_name in data.columns:
            target_col = candidate_col_name
            break
    
    y, x = data[target_col], data.drop(columns=[target_col], axis=1)
    task_type = get_task_type(y)
    if task_type==TaskTypes.REGRESSION:
        automl = run_regression_ensemble(x,y)
    else:
        automl = run_classification_ensemble(x,y)
    print(automl.leaderboard())
    return task_type

for folder in data_folder_names:
    if folder=='bnp-paribas':
        continue
    try:
        folder_contents = os.listdir(f'data/{folder}')
        is_uci_dataset = any('.names' in f for f in folder_contents)
        if is_uci_dataset:
            results = process_uci_dataset(folder)
        else:
            results = process_kaggle_dataset(folder)
        
        print(folder, results)
    except NotImplementedError as e:
        print(f'Skipped {folder}')