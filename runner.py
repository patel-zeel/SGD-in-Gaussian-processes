import os
from config import Config as cfg # Common configuration

# bike
os.system(cfg.abs_path+'preprocess.py bike')