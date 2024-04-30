# like pipeline v5 but with *class conditioning*!   ("wow" <- you)

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
from datetime import datetime

def main(train=True):
    base_dir = os.getcwd()

    now = datetime.now()
    now_str = now.strftime("%d-%m_%H:%M")
    dir_name = f'imagesv6_{now_str}'

    if not os.path.exists(dir_name):
        # If the directory doesn't exist, create it
        os.makedirs(dir_name) 

    #---------------Load VAE and AE models-------------------
    sketch_auto_encoder = torch.load('sketch_AE_v1_4classes.pth')
    photo_var_auto_encoder = torch.load('VAE_v1_4classes_conditioned_with_label.pth') #'VAE_v1_4classes.pth')
    print("Models loaded successfully.")
    #----------------Load dataset---------------------------
    transform = transforms.Compose([transforms.ToTensor()])
    classes = ['tiger', 'dog', 'cat', 'zebra']
    SketchySet = get_sketchimageclass(*classes, transform=transform, size=256, rgb=False, label=True)
    #SketchySet = create_smaller_dataset(SketchySet, 100)
    dataset_size = len(SketchySet)
    train_size = int(0.95 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(SketchySet, [train_size, val_size])
    #----------------------Load UNet model-------------------
    unet = smallSketchUNet(
        image_size=32, # latent size
        in_channels=4,
        out_channels=3,
        num_classes=len(classes)
    )
    print("unet loaded successfully.")
    lr = 0.005
    #----------------------Multi GPU setup-------------------
    ngpus = torch.cuda.device_count()
    batch_size = 80

    diffusion_steps = 1000
    sampling_steps = 250
    DDIM = True

    batch_size = batch_size // ngpus
    torch.distributed.init_process_group("nccl")#(backend="nccl", init_method="env://")
    local_rank= torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet=unet.to(device)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)
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
    diffusion = GuassianDiffusion(diffusion_steps, device = device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=59, gamma=0.7)

    N_EPOCHS = 1000
    if train:
        Tloss=[]
        Vloss = []
        for epoch in tqdm(range(N_EPOCHS)):
            train_loss = 0
            val_loss = 0
            if sampler is not None:
                sampler.set_epoch(epoch)
            if sampler2 is not None:
                sampler2.set_epoch(epoch)
            i=0
            for sketch, photo, label in train_loader:
                #print("new batch")
                if i % 20 == 0 or i==len(train_loader)-1:
                    print("batch number: ", i)
                assert (photo.max().item() <= 1) and (0 <= photo.min().item())
                photo = photo.to(device)
                sketch = sketch.to(device)
                label = torch.LongTensor([classes.index(s) for s in label]).to(device) # label is a list
                #print("recieved photo")
                #photo, sketch = (2 * photo - 1, sketch)
                encoded_photo = photo_var_auto_encoder.encoder(photo).detach()
                #print("recieved encoded photo")
                mu = photo_var_auto_encoder.mu(encoded_photo).detach()
                logvar = photo_var_auto_encoder.logvar(encoded_photo).detach()
                latent_photo = photo_var_auto_encoder.reparameterize(mu,logvar)
                lab=photo_var_auto_encoder.embed(label).detach()
                lab=lab.view(latent_photo.shape[0],latent_photo.shape[1],1,1).detach()
                latent_photo=latent_photo + lab
                #print("recieved latent photo")
                t = torch.randint(diffusion.timesteps, (len(latent_photo),), dtype=torch.int64).to(device)
                #print("recieved t")
                xt, eps = diffusion.sample_from_forward_process(latent_photo, t)
                latent_sketch = sketch_auto_encoder.encoder(sketch).detach()
                #print("recieved latent sketch")
                pred_eps = unet(xt, t, y = label, sketch=latent_sketch)
                #print("recieved pred_eps")
                loss = ((pred_eps - eps) ** 2).mean() #+ reconstruction_loss
                #print("recieved loss")
                train_loss+=loss.item()
                loss.backward()
                optimizer.step()
                #print("recieved optimizer")
                optimizer.zero_grad()
                i=i+1

            train_loss=train_loss/len(train_loader)
            Tloss.append(train_loss)

            scheduler.step()

            j=0
            for sketch, photo, label in val_loader:
                photo = photo.to(device)
                sketch = sketch.to(device)
                label = torch.LongTensor([classes.index(s) for s in label]).to(device)
                if j%20 ==0:
                    print("validation batch number: ", j)
                #pdb.set_trace()
                encoded_photo = photo_var_auto_encoder.encoder(photo).detach()
                #print("recieved encoded photo")
                mu = photo_var_auto_encoder.mu(encoded_photo).detach()
                logvar = photo_var_auto_encoder.logvar(encoded_photo).detach()
                #print("recieved mu and logvar")
                latent_photo = photo_var_auto_encoder.reparameterize(mu,logvar)
                #print("recieved latent photo")
                lab=photo_var_auto_encoder.embed(label)
                lab=lab.view(latent_photo.shape[0],latent_photo.shape[1],1,1)
                latent_photo=latent_photo + lab
                t = torch.randint(diffusion.timesteps, (len(latent_photo),), dtype=torch.int64).to(device)
                xt, eps = diffusion.sample_from_forward_process(latent_photo, t)
                latent_sketch = sketch_auto_encoder.encoder(sketch).detach()

                pred_eps = unet(xt, t, y = label, sketch=latent_sketch)
                loss = ((pred_eps - eps) ** 2).mean()
                val_loss += loss.item()
                j=j+1

            val_loss=val_loss/len(val_loader)
            Vloss.append(val_loss)
            print("Validation loss calculated")
            if epoch % 10 == 0:
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
                print('Epoch: {} \tvalidation Loss: {:.6f}'.format(epoch, val_loss))
                torch.save(unet, f'{dir_name}/intermediate_model.pth')
        for param_group in optimizer.param_groups:
            print("Current learning rate is: ", param_group['lr'])

        unetname= f'{dir_name}/final_modelV6.pth'
        torch.save(unet.module, unetname)
        plt.figure()
        plt.plot(range(1,len(Tloss)+1), Tloss, label='Training loss')
        plt.plot(range(1,len(Vloss)+1), Vloss, label='Validation loss')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.savefig(f"{dir_name}/loss_plot_v6.png")
        plt.close()
    else:
        unet = torch.load(f'{dir_name}/final_modelV6.pth')

    sketch, photo, label = next((iter(train_loader)))
    #images, sketches = (2 * images.to(device) - 1, sketches.to(device))
    photo = photo.to(device)
    sketch = sketch.to(device)
    label = torch.LongTensor([classes.index(s) for s in label]).to(device)

    latent_sketches = sketch_auto_encoder.encoder(sketch).detach()

    encoded_photo = photo_var_auto_encoder.encoder(photo).detach()
    mu = photo_var_auto_encoder.mu(encoded_photo).detach()
    logvar = photo_var_auto_encoder.logvar(encoded_photo).detach()
    latent_photo = photo_var_auto_encoder.reparameterize(mu, logvar)
    lab=photo_var_auto_encoder.embed(label)
    lab=lab.view(latent_photo.shape[0],latent_photo.shape[1],1,1)
    latent_photo=latent_photo + lab

    unet.eval()

    t = torch.randint(diffusion.timesteps, (len(latent_photo),), dtype=torch.int64).to(
        device
    )

    xt, eps = diffusion.sample_from_forward_process(latent_photo, t) #diffusion.timesteps - 1)

    print(f"=== Sampling reversed with {diffusion.timesteps} steps. ===")
    #pdb.set_trace()
    pred_eps=diffusion.sample_from_reverse_process(unet, xt, diffusion.timesteps, model_kwargs={'sketch':latent_sketches, 'y':label}, ddim=DDIM)

    final_gen = photo_var_auto_encoder.decoder(pred_eps)

    pred_eps=(pred_eps.to(device) + 1) / 2
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

        plt.savefig(f"{dir_name}/trained_model_demo_{i}.png")

if __name__ == '__main__':
    main(train=True)
