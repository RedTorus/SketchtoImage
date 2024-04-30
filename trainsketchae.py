from diffmain import *
from DataL import *
from torchsummary import summary
from simpleAE import myAE
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


seed = 112233

#-----------------------Load Data---------------------------------------
transform = transforms.ToTensor()
SketchySet = get_imageset("tiger", "cat", "dog", "zebra", transform = transform, image = False, rgb = False, size = 256)

dataset_size = len(SketchySet)
train_size = int(0.95 * dataset_size)
val_size = dataset_size - train_size

# Splitting the dataset
train_dataset, val_dataset = random_split(SketchySet, [train_size, val_size])

#------------------------Set Multiple GPUs------------------------------------
batch_size = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = myAE(1, 1, variation=False, in_channel=1).to(device)


train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, pin_memory=True)

n_epochs = 10
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.6)  # Decays the learning rate by a factor of 0.1 every 30 epochs

train_losses = []
val_losses = []
for epoch in tqdm(range(n_epochs)):
    model.train()
    train_loss = 0
    val_loss = 0

    for image, sketch in train_loader:
        optimizer.zero_grad()

        sketch = sketch.to(device)
        decoded = model(sketch)

        loss = criterion(decoded, sketch)#torch.mean(model.module.loss_function(out, image, mu, logvar))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    for image, sketch in val_loader:
        image = image.to(device)
        sketch = sketch.to(device)
        
        decoded = model(sketch)
        
        loss = criterion(decoded, sketch) #torch.mean(model.module.loss_function(val_out, image, val_mu, val_logvar))
        val_loss += loss.item()
        
    scheduler.step()
    train_loss = train_loss/len(train_loader)
    val_loss = val_loss/len(val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if epoch % 10 == 0:
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        print('Epoch: {} \tvalidation Loss: {:.6f}'.format(epoch, val_loss))

plt.figure()
plt.plot(range(1,len(train_losses)+1), train_losses, label='Training loss')
plt.plot(range(1,len(val_losses)+1), val_losses, label='validation loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training & Validation Loss")
plt.legend()
#plt.show()
plt.savefig("sketchae_V1_loss_plot.png")

plt.close()


image, sketch = next(iter(train_loader))
sketch = sketch[4].unsqueeze(0)

sketch = sketch.to(device)
print(sketch.size())
x_hat = model(sketch)
print(x_hat.size())

sketch = sketch.cpu().detach().numpy()
x_hat = x_hat.cpu().detach().numpy()

fig, ax = plt.subplots(1,2,figsize=(10,5))
sketch = sketch.squeeze(0).transpose((1,2,0))
x_hat = x_hat.squeeze(0).transpose((1,2,0))
ax[0].imshow(sketch, cmap='gray')
ax[0].set_title('Sketch')
ax[0].axis('off')

ax[1].imshow(x_hat, cmap='gray')
ax[1].set_title('image')
ax[1].axis('off')
plt.savefig("sketchae_V1_beforeafter_plot.png")

plt.close()


torch.save(model, "sketch_AE_v1_4classes.pth")

