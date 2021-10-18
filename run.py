# Imports
import sys
import torch
import pandas as pd
from varz.torch import Vars
from config import Config as cfg # Common configuration
from functions import objective, batch_xy

# Arguments
data_path = sys.argv[1]
method = sys.argv[2]

# Load x and y data
x_train = torch.tensor(pd.read_csv(data_path+'/x_train.csv'))
x_test = torch.tensor(pd.read_csv(data_path+'/x_test.csv'))
y_train = torch.tensor(pd.read_csv(data_path+'/y_train.csv'))
y_test = torch.tensor(pd.read_csv(data_path+'/y_test.csv'))

# Initialize trainable variables
torch.manual_seed(cfg.seed)

vs = Vars(cfg.dtype)

vs.positive(name='variance')
vs.positive(name='length_scale')
vs.positive(name='noise')
vs.requires_grad(True) # Torch specific

# Training
if method == 'sgd':
    optimizer = torch.optim.SGD(vs.get_latent_vector(), lr=cfg.lr)
    for epoch in range(1, cfg.epochs+1):
        x_batch, y_batch = batch_xy(x_train, y_train, method=cfg.method)
        
        optimizer.param_groups[0]['lr'] = cfg.lr/epoch
        optimizer.zero_grad()
        loss = objective(vs, x, y)
        print(loss.item())
        loss.backward()
        optimizer.step()