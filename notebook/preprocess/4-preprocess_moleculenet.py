import deepchem as dc
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
import warnings

warnings.simplefilter('ignore')
import pandas as pd

from mpi4py import MPI
from tqdm import tqdm

usable = set([method for method in dir(dc.molnet) if "load_" in method])

not_usable = set([
    "load_bandgap",
    "load_kinase",
    "load_perovskite",
    "load_uv",
    "load_kaggle",
    "load_function",
    "load_factors",
    "load_mp_formation_energy",
    "load_pdbbind",
    "load_mp_metallicity",
    "load_Platinum_Adsorption",
    "load_bbbc001",
    "load_bbbc002",
    "load_bbbc003",
    "load_bbbc004",
    "load_bbbc005",
    "load_cell_counting",
    "load_uspto",
    "load_zinc15",
    "load_chembl25"
])

# These load_* functions in not_usable represent datasets where the input features (X) either do not contain SMILES strings or do not correspond to a single compound's SMILES representation.
# load_zinc15 and load_chembl25 were abandoned because the dateset was too large.

usable = usable - not_usable

for i in range(len(usable)):
    function_name = usable[i]
    dataset_name = function_name.replace("load_", "")
    tasks, datasets, transformers = eval('dc.molnet.' + function_name + '(featurizer="GraphConv")')

    train, test, valid = datasets
    Y = np.concatenate((train.y, test.y, valid.y), axis=0)

    tsv = []
    for x in [train, test, valid]:
        for u in range(len(x)):
            try:
                old = x.ids[u]
                mol = Chem.MolFromSmiles(old)
                new = Chem.MolToSmiles(mol)
                col = []
                col.append(new)
                for i in range(len(Y[u])):
                    col.append(Y[u][i])
                tsv.append(col)
            except:
                continue

    os.makedirs(f"../../data/processed/moleculenet/moleculenet_patentdate/{dataset_name}", exist_ok=True)
    pd.DataFrame(tsv).to_csv(f"../../data/processed/moleculenet/moleculenet_patentdate/{dataset_name}/pre_mol_dataset.tsv", sep="\t", header=None, index=False)

    with open(f"../../data/processed/moleculenet/moleculenet_patentdate/{dataset_name}/tasks.txt", "a") as f:
        for task in tasks:
            f.write(task+"\n")

sc_smiles_date = pickle_load("../../data/processed/surechembl/250106_surechembl_smiles_date.pickle")
smiles_sc = set(sc_smiles_date.keys())
paths = glob.glob("../../data/processed/moleculenet/moleculenet_patentdate/*/pre_mol_dataset.tsv")

for path in paths:
    output_path = path.replace("/pre_mol_dataset.tsv", "") + "/date_dataset.tsv"
    df = pd.read_csv(path, sep="\t", header=None)
    smiles = df[0]
    tsv = []
    for i in range(len(smiles)):
        s = smiles[i]
        if s not in smiles_sc:
            continue
        else:
            date = sc_smiles_date[s].replace("-", "")
            date = convert_date_to_decimal(date)
            col = []
            col.append(s)
            col.append(date)
            for n in range(1, len(df.iloc[i])):
                col.append(df.iloc[i,n])
            tsv.append(col)
    pd.DataFrame(tsv).to_csv(output_path, sep="\t", header=None, index=False)

for path in paths:
    task_path = path.replace("/pre_mol_dataset.tsv", "") + "/tasks.txt"
    with open(task_path, "r") as f:
        tasks = f.readlines()
    date_path = path.replace("/pre_mol_dataset.tsv", "")  + "/date_dataset.tsv"
    df = pd.read_csv(date_path, sep="\t", header=None)
    df = df.sort_values(1, ignore_index=True)

    ecfp = []
    smiles = df[0]
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        ecfp.append(fp)

    for n in range(len(tasks)):
        task = tasks[n].replace("\n", "").replace("/", "_")
        y = df[n+2]
        tsv = []
        for m in range(len(y)):
            tsv.append([ecfp[m], y[m]])
        pickle_dump(tsv, f"../../data/processed/moleculenet/moleculenet_patentdate/{dataset_name}/{task}_training.pickle")

date_paths = "../../data/processed/moleculenet/moleculenet_patentdate/*/date_dataset.tsv"
output_file = "../../data/processed/moleculenet/details.tsv"

for i in range(len(date_paths)):
    date_path = date_paths[i]
    pre_path = date_path.replace("_date", "")
    class_name = date_path.split("/")[1]
    test_name = date_path.split("/")[2].replace("_date.tsv", "")

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
        f.write("\t".join(map(str, [class_name, test_name, l_date, l_pre, per_date, task, date_score, pre_score])) + "\n")