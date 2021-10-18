import torch
from stheno import GP, EQ

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import Config as cfg # Common configuration

def model(vs):
    return vs['variance']*GP(EQ().stretch(vs['length_scale']))

def objective(vs, x, y):
    gp = model(vs)
    return -gp(x, vs['noise']).logpdf(y)

def batch_xy(x, y, size, method):
    if method == 'uniform':
        inds = torch.multinomial(torch.arange(x.shape[0]), num_samples=size)
        return x[inds], y[inds]

def split_and_save(x, y, train_size, seed, name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=seed)

    x_train.to_csv(cfg.abs_path+'final_data/'+name+'/x_train.csv', index=None)


