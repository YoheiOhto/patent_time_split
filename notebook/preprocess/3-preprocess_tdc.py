
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
import warnings

warnings.simplefilter('ignore')
import glob
import os
import sys
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
from tdc import multi_pred, single_pred
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list
from tqdm import tqdm

current_dir = os.getcwd()
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(parent_parent_dir, 'src')
sys.path.append(src_dir)

from util import *

## Following https://tdcommons.ai/single_pred_tasks/tox and the example below, download tdc's ADME, Tox, HTS dataset. 
## toxcast and tox21 are contained in moleculenet. so, they are treated in molculenet.

# label_list = retrieve_label_name_list('herg_central')
# for label in label_list:
#     data = Tox(name = 'herg_central', label_name = label)
#     splits = data.get_split()

#     rdkit_smiles = []
#     ys= []
#     for n in splits:
#         smiles = splits[n]["Drug"]
#         y = splits[n]["Y"]
#         for i in tqdm(range(len(smiles))):
#             old = smiles[i]
#             mol = Chem.MolFromSmiles(old)
#             new = Chem.MolToSmiles(mol)
#             rdkit_smiles.append(new)
#             ys.append(y[i])
        
#     output = f"../../data/processed/tdc/Tox/{label}.tsv"
#     tsv = []
#     for i in range(len(rdkit_smiles)):
#         tsv.append([rdkit_smiles[i], ys[i]])
#     pd.DataFrame(tsv).to_csv(output, sep="\t")


sc_smiles_date = pickle_load("../../data/processed/surechembl/250106_surechembl_smiles_date.pickle")

dir_paths = ["../../data/processed/tdc/Tox/", "../../data/processed/tdc/ADME/", "../../data/processed/tdc/HTS/"]

for dir_path in dir_paths:
    paths = glob.glob(dir_path + "*.tsv")
    for path in paths:
        print(path)
        output_file = path.replace(".tsv", "") + "_date.tsv"
        df = pd.read_csv(path, sep="\t", index_col=0)
        
        tsv = []
        for i in range(len(df)):
            smiles = df.iloc[i,0]
            tox = df.iloc[i,1]
            try:
                date = sc_smiles_date[smiles]
                tsv.append([smiles, tox, date])
            except:
                continue
        pd.DataFrame(tsv).to_csv(output_file, sep="\t", index=False)

dir_paths = ["../../data/processed/tdc/Tox/", "../../data/processed/tdc/ADME/", "../../data/processed/tdc/HTS/"]

for dir_path in dir_paths:
    paths = glob.glob(dir_path + "*_date.tsv")
    for path in paths:
        task = path.replace(dir_path, "").replace("_date.tsv", "")
        print(task)

        df = pd.read_csv(path, sep="\t")
        df = df.sort_values("2", ignore_index=True)

        tsv = []
        for i in range(len(df)):
            s = df.iloc[i,0]
            y = df.iloc[i,1]
            mol = Chem.MolFromSmiles(s)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            tsv.append([fp, y])
        pickle_dump(tsv, f"{dir_path}/{task}_training.pickle")

date_paths = glob.glob("../../data/processed/tdc/*/12_hERG*_date.tsv")
output_file = "../../data/processed/tdc/details.tsv"
# open(output_file, "w").close()
print(date_paths)

for i in range(len(date_paths)):
    date_path = date_paths[i]
    pre_path = date_path.replace("_date", "")
    class_name = date_path.split("/")[5]
    test_name = date_path.split("/")[6].replace("_date.tsv", "")
    print(class_name, test_name)

    df_date = pd.read_csv(date_path, sep="\t")
    df_pre = pd.read_csv(pre_path, sep="\t", index_col=0)

    l_date = len(df_date)
    l_pre = len(df_pre)
    per_date = l_date / l_pre

    y_date = df_date["1"]
    y_pre = df_pre["1"]
    task = predict_task_equal_class(y_date)

    if task == 10000000:
        # "regression"
        date_score = sum(y_date) / l_date
        pre_score = sum(y_date) / l_date
    else:
        if task != 2:
            print(test_name, task)
        date_score = sum(y_date) / l_date
        pre_score = sum(y_pre) / l_pre
    
    with open(output_file, "a") as f:
        f.write("\t".join([str(x) for x in [class_name, test_name, l_date, l_pre, per_date, task, date_score, pre_score]]) + "\n")