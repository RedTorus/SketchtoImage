import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pathlib
from matplotlib import pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, real_dir, sketch_dir, transform=None):
        self.real_dir = real_dir
        self.sketch_dir = sketch_dir
        self.file_names = os.listdir(real_dir)
        self.transform = transform
        self.pairs = []
        for name in self.file_names:
            real_img_name = os.path.join(self.real_dir, name)
            suffix = pathlib.Path(name).suffix
            sketch_img_name = os.path.join(self.sketch_dir, name.removesuffix(suffix)+'.png')
            print()
            real_image = Image.open(real_img_name)
            sketch_image = Image.open(sketch_img_name)

            if self.transform:
                real_image = self.transform(real_image)
                sketch_image = self.transform(sketch_image)

            self.pairs.append((sketch_image, real_image))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# Change directories if needed.
real_dir = "/home/aidan/ComputerScience/school/18786-DeepLearning/project/photosketch/PhotoSketch-master/examples"
sketch_dir = "/home/aidan/ComputerScience/school/18786-DeepLearning/project/photosketch/PhotoSketch-master/results"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def get_dataloader(batch_size=16, shuffle=True):
    # Create dataset and dataloader
    custom_dataset = CustomDataset(real_dir, sketch_dir, transform=transform)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def display_tensor_images(tensor_images, title=None, figsize=(10, 5)):
    batch_size, channels, height, width = tensor_images.shape
    tensor_images = tensor_images.permute(0, 2, 3, 1)
    tensor_images = tensor_images.cpu().numpy()

    fig, axes = plt.subplots(nrows=1, ncols=batch_size, figsize=figsize)
    if title:
        fig.suptitle(title)
    for i in range(batch_size):
        axes[i].imshow(tensor_images[i])
        axes[i].axis('off')
    plt.show()

if __name__ == '__main__':
    print(len(dataloader))
    for batch_idx, (sketch_images, real_images) in enumerate(dataloader):
        display_tensor_images(real_images)
        display_tensor_images(sketch_images)
