import torch

class Config:
    abs_path = '/home/patel_zeel/SGD-in-Gaussian-processes/'
    train_size = 0.6
    lr = 0.1
    epochs = 100
    method = 'uniform'
    dtype = torch.float32
    seed = 0
    fold = 0