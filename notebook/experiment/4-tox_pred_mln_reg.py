import os
import sys
import warnings
from datetime import datetime
import glob
import pickle

import pandas as pd
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import (accuracy_score, classification_report, f1_score, r2_score,
                             roc_auc_score, mean_squared_error, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
warnings.simplefilter('ignore')

current_dir = os.getcwd()
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(parent_parent_dir, 'src')
sys.path.append(src_dir)

from util import *

df_details = pd.read_csv("../../data/processed/moleculenet/details.tsv",sep="\t",header=None)
use = ['bace', 'bbbp', 'clintox', 'hiv', 'sider', 'tox21', 'toxcast']

for i in tqdm(range(len(df_details))):
mn_results_paths
    if df_details.iloc[i,0] not in use:
        continue

    fp_date = pickle_load(f"../../data/processed/moleculenet/moleculenet_patentdate/{df_details.iloc[i,0]}/{df_details.iloc[i,1]}_training.pickle")
    X = [x[0] for x in fp_date]
    y = [Y[1] for Y in fp_date]
    
    # time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    result = [task, "time", r2, mse, mae, sum(y)/len(y), sum(y_train)/len(y_train), sum(y_test)/len(y_test)]
    results.append(result)

    for seed in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        result = [task, "random", r2, mse, mae, sum(y)/len(y), sum(y_train)/len(y_train), sum(y_test)/len(y_test)]
        results.append(result)
    pd.DataFrame(results).to_csv(f"../../data/result/moleculenet_scores/{df_details.iloc[i,0]}/moleculenet_30%.tsv", sep="\t", header=None, index=False)

for i in tqdm(range(len(df_details))):
    task = f"{df_details.iloc[i,0]}/{df_details.iloc[i,1]}"
    results = []
    if df_details.iloc[i,5] == 2 or df_details.iloc[i,5] == 1:
        continue
    if df_details.iloc[i,0] not in use:
        continue

    fp_date = pickle_load(f"../../data/processed/moleculenet/moleculenet_patentdate/{df_details.iloc[i,0]}/{df_details.iloc[i,1]}_training.pickle")
    X = [x[0] for x in fp_date]
    y = [Y[1] for Y in fp_date]
    
    # time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    result = [task, "time", r2, mse, mae, sum(y)/len(y), sum(y_train)/len(y_train), sum(y_test)/len(y_test)]
    results.append(result)

    for seed in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        result = [task, "random", r2, mse, mae, sum(y)/len(y), sum(y_train)/len(y_train), sum(y_test)/len(y_test)]
        results.append(result)
    pd.DataFrame(results).to_csv(f"../../data/result/moleculenet_scores/{df_details.iloc[i,0]}/moleculenet_20%.tsv", sep="\t", header=None, index=False)