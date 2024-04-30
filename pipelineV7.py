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
from overall_model import SketchToImage
import multiprocessing

def main(train=True):
    base_dir = os.getcwd()
    #---------------Load VAE and AE models-------------------
    AE = torch.load('sketch_AE_v1_4classes.pth')
    VAE = torch.load('VAE_v1_4classes.pth')
    print("Models loaded successfully.")
    #----------------Load dataset---------------------------
    transform = transforms.Compose([transforms.ToTensor()])
    SketchySet = get_sketchimageclass('tiger', 'dog', 'cat', 'zebra', transform=transform, size=256, rgb=False)
    #SketchySet = create_smaller_dataset(SketchySet, 100)
    dataset_size = len(SketchySet)
    train_size = int(0.05 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(SketchySet, [train_size, val_size])
    #----------------------Load UNet model-------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = smallSketchUNet(
        image_size=32, # latent size
        in_channels=4,
        out_channels=3)
    print("unet loaded successfully.")
    lr = 0.005
    diffusion_steps = 1000
    sampling_steps = 250
    DDIM = True
    diffusion = GuassianDiffusion(diffusion_steps, device = device)
    #----------------------Multi GPU setup-------------------
    ngpus = torch.cuda.device_count()
    batch_size = 160

    batch_size = batch_size // ngpus
    torch.distributed.init_process_group("nccl")#(backend="nccl", init_method="env://")
    local_rank= torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    
    sketch_to_image = SketchToImage(AE, VAE, unet, device = "cuda", ddim = True, timesteps=diffusion_steps)
    sketch_to_image=sketch_to_image.to(device)

    sketch_to_image = DDP(sketch_to_image, device_ids=[local_rank], output_device=local_rank)
    sampler = DistributedSampler(train_dataset) if ngpus > 1 else None

    cpu_count = multiprocessing.cpu_count() // ngpus #don't use all cpus else slow min 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=cpu_count, pin_memory=True)
    sampler2 = DistributedSampler(val_dataset) if ngpus > 1 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler2, num_workers=1, pin_memory=True)

    if local_rank == 0:
        print(f"Using distributed training on {ngpus} gpus.")
        print(
            f"Training dataset loaded: Number of batches: {len(train_loader)}, Number of images: {len(SketchySet)}"
        )
    #----------------------------Training Loop---------------------
    

    optimizer = torch.optim.AdamW(sketch_to_image.parameters(), lr=0.07)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    N_EPOCHS = 2
    if train:
        Tloss=[]
        Vloss = []
        sketch_to_image.train()
        for epoch in tqdm(range(N_EPOCHS)):
            train_loss = 0
            val_loss = 0

            if sampler is not None:
                sampler.set_epoch(epoch)
            if sampler2 is not None:
                sampler2.set_epoch(epoch)
            i=0
            for sketch, photo in train_loader:

                assert (photo.max().item() <= 1) and (0 <= photo.min().item())
                photo = photo.to(device)
                sketch = sketch.to(device)

                pred_eps, eps = sketch_to_image(photo, sketch)

                loss = ((pred_eps - eps) ** 2).mean() #+ reconstruction_loss
                train_loss+=loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss=train_loss/len(train_loader)
            Tloss.append(train_loss)

            scheduler.step()

            for sketch, photo in val_loader:
                photo = photo.to(device)
                sketch = sketch.to(device)

                pred_eps, eps = sketch_to_image(photo, sketch)
                loss = ((pred_eps - eps) ** 2).mean()
                val_loss += loss.item()
            
            val_loss=val_loss/len(val_loader)
            Vloss.append(val_loss)

        if epoch % 10 == 0:
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
            print('Epoch: {} \tvalidation Loss: {:.6f}'.format(epoch, val_loss))
            torch.save(sketch_to_image.module, 'intermediate_model.pth')
        for param_group in optimizer.param_groups:
            print("Current learning rate is: ", param_group['lr'])
        
        torch.save(sketch_to_image.module, 'final_model_V7.pth')
        plt.figure()
        plt.plot(range(1,len(Tloss)+1), Tloss, label='Training loss')
        plt.plot(range(1,len(Vloss)+1), Vloss, label='Validation loss')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.savefig("images/loss_plot.png")
        plt.close()
    else:
        sketch_to_image = torch.load('final_model_V7.pth')

#----------------------------Forward pass test--------------------------
    sketch, photo = next((iter(train_loader)))
    #images, sketches = (2 * images.to(device) - 1, sketches.to(device))
    photo = photo.to(device)
    sketch = sketch.to(device)


    sketch_to_image.eval()

    final_gen = sketch_to_image(sketch)

    for i in range(batch_size):
        fig, ax = plt.subplots(1,3,figsize=(10,5))
        ax[0].imshow(photo[i].cpu().permute(1, 2, 0).detach().numpy())
        ax[0].set_title('before image')
        ax[0].axis('off')

        ax[1].imshow(sketch[i].cpu().permute(1, 2, 0).detach().numpy(), cmap='gray')
        ax[1].set_title('sketch')
        ax[1].axis('off')

        ax[2].imshow(final_gen[i].cpu().permute(1, 2, 0).detach().numpy())
        ax[2].set_title('after')
        ax[2].axis('off')

        plt.savefig(f"imagesv7/trained_model_demo_{i}.png")


if __name__ == '__main__':
    main(train=True)
