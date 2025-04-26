import multiprocessing as mp
import os
import pickle
import random
import re
import sys

import numpy as np
import pandas as pd
import tmap
import tmap as tm
from faerun import Faerun
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm
from tqdm.auto import trange

RDLogger.DisableLog('rdApp.*')

import collections
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2

current_dir = os.getcwd()
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(parent_parent_dir, 'src')
sys.path.append(src_dir)

from util import *

enc = tm.Minhash(64)
lf = tm.LSHForest(64)

d = 64
n = 23320261

data = []
input_path = "../../data/processed/surechembl/minhash_vectors.pkl" 

with open(input_path, "rb") as f:
    for _ in trange(n):
        try:
            vec = pickle.load(f)
            converted_vec = [np.uint8(x) for x in vec[0]]
            data.append(converted_vec)
        except EOFError:
            break

data = enc.batch_from_binary_array(data)
lf.batch_add(data)
lf.index()
print("! DONE lsh setup")

knng_from, knng_to, knng_weight = tm.VectorUint(), tm.VectorUint(), tm.VectorFloat()
_ = lf.get_knn_graph(knng_from, knng_to, knng_weight, 10)

print("! DONE knn setup")

import gc
del data
del knng_from
del knng_to
del knng_weight
gc.collect()

x, y, s, t, _ = tm.layout_from_lsh_forest(lf)

print("! DONE tmap layout setting")

output_dir = "../../data/result/tmap/surechembl/whole"
os.makedirs(output_dir, exist_ok=True)

filepaths = {
    "x": f"{output_dir}/x.pkl",
    "y": f"{output_dir}/y.pkl",
    "s": f"{output_dir}/s.pkl",
    "t": f"{output_dir}/t.pkl",
}

pickle_dump(list(x), filepaths["x"])
pickle_dump(list(y), filepaths["y"])
pickle_dump(list(s), filepaths["s"])
pickle_dump(list(t), filepaths["t"])

print("! DONE tmap layout output")