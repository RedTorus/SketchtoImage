from diffmain import *
from DataL import SketchyDataset
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
import random

photos_dir = '/Users/deniskaanalpay/Desktop/CMU/Term_2/18786_Intro_to_DL/SketchtoImage/rendered_256x256/256x256/photo/tx_000100000000'
sketches_dir = '/Users/deniskaanalpay/Desktop/CMU/Term_2/18786_Intro_to_DL/SketchtoImage/rendered_256x256/256x256/sketch/tx_000100000000'

print(os.listdir(sketches_dir))
SketchySet = SketchyDataset(sketches_dir, photos_dir)

print(os.listdir(sketches_dir))
SketchySet = SketchyDataset(photos_dir, sketches_dir))

#SketchySet = get_dataset(photos_dir, sketches_dir)


train_loader = torch.utils.data.DataLoader(SketchySet, batch_size=256)


i=random.randint(0,len(SketchySet)-1)
sk,im = SketchySet[i]

# Convert the tensors back to numpy arrays
#sk = sk.numpy()
#im = im.numpy()

sk=np.array(sk)#.transpose((1,2,0))
im=np.array(im)#.transpose((1,2,0))
print(im)
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(sk)
ax[0].set_title('Sketch')
ax[0].axis('off')

ax[1].imshow(im)
ax[1].set_title('image')
ax[1].axis('off')

plt.show()

