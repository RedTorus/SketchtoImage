from diffmain import *
from DataL import *
import glob
import sys
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pdb
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from sketch_condition_unet import SketchUNet, smallSketchUNet
from simpleAE import myAE
#from vae import VAE
import multiprocessing

base_dir = os.getcwd()
#---------------Load VAE and AE models-------------------
sketch_auto_encoder = torch.load('sketch_AE_v1_4classes.pth')
photo_var_auto_encoder = torch.load('VAE_v1_4classes.pth')
print("Models loaded successfully.")
#----------------------Multi GPU setup-------------------
ngpus = torch.cuda.device_count()
batch_size = 160

diffusion_steps = 500
sampling_steps = 250
DDIM = True

batch_size = batch_size // ngpus
torch.distributed.init_process_group("nccl")
local_rank= torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

unet= torch.load("final_modelV5.pth")
unet=unet.to(device)

model = unet.module.to("cpu")
#print(model)
torch.save(model, "final_modelV4_non_distributed.pth")

model_check = torch.load("final_modelV4_non_distributed.pth", map_location='cpu')

