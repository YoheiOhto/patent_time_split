import os
import pickle
from datetime import datetime
from logging import DEBUG, Formatter, getLogger, handlers
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import robust_scale


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

def set_logger():
    root_logger = getLogger()
    root_logger.setLevel(DEBUG)
    rotating_handler = handlers.RotatingFileHandler(
        r'../../log/app.log',
        mode="a",
        maxBytes=100 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    format = Formatter('%(asctime)s : %(levelname)s : %(filename)s - %(message)s')
    rotating_handler.setFormatter(format)
    root_logger.addHandler(rotating_handler)

def file_checker(path, overwrite = False):
    path_obj = Path(path)
    if path_obj.exists():
        if overwrite:
            print(f"{path} is already exists, but OVERWRITE!")
            return True
        else:
            print(f"{path} is already exists!")
            return False
    else:
        return True

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def unzip_all_files_in_directory(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".zip"):
            file_path = os.path.join(input_directory, filename)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(output_directory)
                print(f"Extracted: {filename}")

def convert_date_to_decimal(date_str):
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    leap_year = is_leap_year(year)
    days_in_year = 366 if leap_year else 365
    day_of_year = (date_obj - datetime(year, 1, 1)).days + 1
    progress = day_of_year / days_in_year
    return round(year + progress, 3)

def predict_task_equal_class(y):
    unique_values = np.unique(y)
    if all(isinstance(num, np.int64) or (isinstance(num, np.float64) and num.is_integer()) for num in unique_values):
        return len(unique_values) 
    else:
        return 10000000