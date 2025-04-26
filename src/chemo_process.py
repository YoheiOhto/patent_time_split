import re

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

import warnings

import pubchempy as pcp
from tqdm import tqdm

warnings.simplefilter('ignore')

def process_smiles(args):
    old = args
    try:
        mol = Chem.MolFromSmiles(old)
        if mol is None:
            return None
        new = Chem.MolToSmiles(mol)
        return new, mol
    except:
        return None

def process_smiles_date(args):
    old, date_raw = args
    try:
        mol = Chem.MolFromSmiles(old)
        if mol is None:
            return None
        new = Chem.MolToSmiles(mol)
        date = date_raw.replace("-", "")
        date_decimal = convert_date_to_decimal(date)
        return new, mol, date_decimal
    except:
        return None

def remove_isotopes(smiles_dict):
    data = []
    for smiles, date in tqdm(smiles_dict.items()):
        cleaned_smiles = re.sub(r'\[\d+([A-Za-z]+)[+-]?[0-9]*\]', r'[\1]', smiles)
        data.append([smiles, cleaned_smiles, date])
    return data