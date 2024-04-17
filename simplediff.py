import os
import cv2
import copy
import math
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from easydict import EasyDict
import random
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from data import get_metadata, get_dataset, fix_legacy_dict
import unets
import diffmain
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from diffmain import GuassianDiffusion, loss_logger, train_one_epoch
from DataL import get_miniset


def create_smaller_dataset(original_dataset, factor = 1.0):
    # Calculate the size of the smaller dataset
    smaller_size = len(original_dataset) // factor

    # Randomly select indices for the smaller dataset
    smaller_indices = random.sample(range(len(original_dataset)), smaller_size)

    # Create the smaller dataset
    smaller_dataset = [original_dataset[i] for i in smaller_indices]

    return smaller_dataset

def plot_first_png(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out all non-png files
    png_files = [file for file in files if file.endswith('.jpg')]
    
    # If there are no png files in the directory
    if not png_files:
        print("No PNG files found in the directory.")
        return
    
    # Sort the list of png files
    png_files.sort()
    
    # Get the first png file
    first_png = png_files[0]
    
    # Create the full file path
    file_path = os.path.join(directory, first_png)
    
    # Read the image file
    img = mpimg.imread(file_path)
    
    # Plot the image
    plt.imshow(img)
    plt.show()
    print("Image shape: ", img.shape)

def main():
    parser = argparse.ArgumentParser("Minimal implementation of diffusion models")
    # diffusion model
    parser.add_argument("--arch", type=str, default="UNet", help="Neural network architecture")
    parser.add_argument(
        "--class-cond",
        action="store_true",
        default=False,
        help="train class-conditioned diffusion model",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=250,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Sampling using DDIM update step",
    )
    # dataset
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-dir", type=str, default="./rendered_256x256/256x256/sketch/resized/ant")
    # optimizer
    parser.add_argument(
        "--batch-size", type=int, default=2, help="batch-size per gpu"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ema_w", type=float, default=0.9995)
    # sampling/finetuning
    parser.add_argument("--pretrained-ckpt", type=str, help="Pretrained model ckpt")
    parser.add_argument("--delete-keys", nargs="+", help="Pretrained model ckpt")
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        default=False,
        help="No training, just sample images (will save them in --save-dir)",
    )
    parser.add_argument(
        "--num-sampled-images",
        type=int,
        default=50000,
        help="Number of images required to sample from the model",
    )

    # misc
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)

    # setup
    args = parser.parse_args()
    #metadata = get_metadata(args.dataset)
    torch.backends.cudnn.benchmark = True
    args.device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    if args.local_rank == 0:
        print(args)

    print("device: ", args.device)

    plot_first_png(args.data_dir)

    model = unets.__dict__[args.arch](
        image_size=128,
        in_channels=3,
        out_channels=3,
        num_classes=2 if args.class_cond else None,
    ).to(args.device)

    if args.local_rank == 0:
        print(
            "We are assuming that model input/ouput pixel range is [-1, 1]. Please adhere to it."
        )

    diffusion = GuassianDiffusion(args.diffusion_steps, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # distributed training
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        if args.local_rank == 0:
            print(f"Using distributed training on {ngpus} gpus.")

        args.batch_size = args.batch_size // ngpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    train_set = get_miniset()#get_dataset(args.dataset, args.data_dir, metadata)
    train_set = create_smaller_dataset(train_set)

    sampler = DistributedSampler(train_set) if ngpus > 1 else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
    )

    #print("EMA", args.ema_dict)
    if args.local_rank == 0:
        print(
            f"Training dataset loaded: Number of batches: {len(train_loader)}, Number of images: {len(train_set)}"
        )

    logger = loss_logger(len(train_loader) * args.epochs)
    
    train_one_epoch(model, train_loader, diffusion, optimizer, logger, None, args)
    print("Training done")

if __name__ == "__main__":
    main()