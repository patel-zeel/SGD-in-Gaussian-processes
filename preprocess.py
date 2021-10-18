import sys
from config import Config as cfg # Common configuration
from functions import split_and_save
import pandas as pd
import zipfile

# Arguments
data_name = sys.argv[1]

# Data loading & preprocessing
if data_name == 'bike':
    zip_path = 'raw_data/Bike-Sharing-Dataset.zip'
    file_name = 'hour.csv'
    archive = zipfile.ZipFile(zip_path)
    data = pd.read_csv(archive.open(file_name))

    x_cols = ['dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
        'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    y_col = ['cnt']

    # convert date strings to floats
    data['dteday'] = pd.to_datetime(data['dteday']).astype(int)/ 10**18

split_and_save(data[x_cols], data[y_col], train_size=cfg.train_size, seed=cfg.seed, name=data_name)