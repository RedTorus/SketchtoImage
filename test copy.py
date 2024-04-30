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

unet = torch.load("final_modelV4_non_distributed.pth", map_location='cpu')