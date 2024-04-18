import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms


class OnlyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Walk through the directory tree and collect paths of all .jpg images
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    self.image_paths.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # For your use case, the prediction and target are the same image
        return image, image

class SketchyData(Dataset):
    def __init__(self, sketchy_dir, image_dir, transform=None):
        self.sketchy_dir = sketchy_dir
        self.image_dir = image_dir
        self.transform = transform
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = []
        # Build a dictionary with key as base filename without extension and value as full path
        image_dict = {os.path.splitext(file)[0]: os.path.join(dirpath, file)
                      for dirpath, dirnames, filenames in os.walk(self.image_dir)
                      for file in filenames if file.endswith('.jpg')}
        
        # Iterate over sketch files and match them with corresponding images
        for dirpath, dirnames, filenames in os.walk(self.sketchy_dir):
            for file in filenames:
                if file.endswith('.png'):
                    base_name = re.sub(r'-\d+\.png$', '', file)  # Remove the sketch number and extension
                    if base_name in image_dict:
                        pairs.append((os.path.join(dirpath, file), image_dict[base_name]))
        return pairs


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        
        sketch_path = os.path.join(self.sketchy_dir, self.pairs[idx][0])
        image_path = os.path.join(self.image_dir, self.pairs[idx][1])

        sketch = Image.open(sketch_path).convert('RGB')
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            sketch = self.transform(sketch)
            image = self.transform(image)

        return sketch, image


class miniset(Dataset):
    def __init__(self, png_dir, jpg_dir, transform=None):
        self.png_dir = png_dir
        self.jpg_dir = jpg_dir
        self.transform = transform

        # Create a dictionary mapping from png file names to jpg file names
        self.mapping = {}
        for png_file in os.listdir(png_dir):
            if png_file.endswith('.png'):
                base_name = png_file[:-6]  # Remove the '-n.png' part
                corresponding_jpg_file = base_name + '.jpg'
                if corresponding_jpg_file in os.listdir(jpg_dir):
                    self.mapping[png_file] = corresponding_jpg_file

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        png_file = list(self.mapping.keys())[idx]
        jpg_file = self.mapping[png_file]

        png_path = os.path.join(self.png_dir, png_file)
        jpg_path = os.path.join(self.jpg_dir, jpg_file)

        png_image = Image.open(png_path).convert('RGB')
        jpg_image = Image.open(jpg_path).convert('RGB')

        if self.transform:
            png_image = self.transform(png_image)
            jpg_image = self.transform(jpg_image)

        transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        png_image = transform(png_image)
        jpg_image = transform(jpg_image)

        return jpg_image, png_image #png_image, jpg_image

def resize_images_in_subdirs(input_base_dir, output_base_dir, size):
    # Walk through the input directory
    for dirpath, dirnames, filenames in os.walk(input_base_dir):
        for filename in filenames:
            # Check if the file is a PNG or JPG image
            if filename.endswith('.png') or filename.endswith('.jpg'):
                # Create the full file path for the input image
                input_file_path = os.path.join(dirpath, filename)
                
                # Open the image file
                img = Image.open(input_file_path)
                
                # Resize the image
                img_resized = img.resize(size)
                
                # Create the corresponding output directory
                output_dir = dirpath.replace(input_base_dir, output_base_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Create the full file path for the output image
                output_file_path = os.path.join(output_dir, filename)
                
                # Save the resized image to the output file
                img_resized.save(output_file_path)



def get_dataset():
    base_dir = os.path.dirname(os.path.realpath(__file__))
<<<<<<< HEAD
    sketchpath = os.path.join(base_dir,'256x256/sketch/tx_000100000000')  # \\zebra'
    imagepath = os.path.join(base_dir,'256x256/photo/tx_000100000000')  # \\zebra'
    SketchySet = SketchyDataset(sketches_dir=sketchpath, images_dir=imagepath)
=======
    sketchpath = os.path.join(base_dir,'rendered_256x256/256x256/sketch/tx_000100000000')  # \\zebra'
    imagepath = os.path.join(base_dir,'rendered_256x256/256x256/photo/tx_000100000000')  # \\zebra'
    SketchySet = SketchyData(sketchy_dir=sketchpath, image_dir=imagepath)
>>>>>>> 291c8d9f09079497fcf599a5f3896b97366e40d8

    return SketchySet

def get_miniset():
    base_dir = os.path.dirname(os.path.realpath(__file__))
<<<<<<< HEAD
    png_dir = os.path.join(base_dir, '256x256/sketch/resized/airplane')
    jpg_dir = os.path.join(base_dir, '256x256/photo/resized/airplane')
=======
    png_dir = os.path.join(base_dir, 'rendered_256x256/256x256/sketch/resized/ant')
    jpg_dir = os.path.join(base_dir, 'rendered_256x256/256x256/photo/resized/ant')
>>>>>>> 291c8d9f09079497fcf599a5f3896b97366e40d8
    mini = miniset(png_dir, jpg_dir)

    return mini


def get_imageset():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = os.path.join(base_dir, 'rendered_256x256/256x256/photo/resized')
    image_set = OnlyImageDataset(image_dir)

    return image_set

<<<<<<< HEAD
    resize_images_in_subdirs('256x256/sketch/tx_000100000000', '256x256/sketch/resized', (128, 128))
    resize_images_in_subdirs('256x256/photo/tx_000100000000', '256x256/photo/resized', (128, 128))
=======
#sketchpath='C:\\Users\\kaust\\Downloads\\SketchyDataloader\\rendered_256x256\\256x256\\sketch\\tx_000100000000'#\\zebra'
#imagepath='C:\\Users\\kaust\\Downloads\\SketchyDataloader\\rendered_256x256\\256x256\\photo\\tx_000100000000'#\\zebra'
>>>>>>> 291c8d9f09079497fcf599a5f3896b97366e40d8

#resize_images_in_subdirs('rendered_256x256/256x256/sketch/tx_000100000000', 'rendered_256x256/256x256/sketch/resized', (128, 128))
#resize_images_in_subdirs('rendered_256x256/256x256/photo/tx_000100000000', 'rendered_256x256/256x256/photo/resized', (128, 128))

SketchySet = get_imageset()#get_dataset()#SketchyData(sketchy_dir = sketchpath, image_dir =imagepath)

#print(len(SketchySet.sketchy_dict), len(SketchySet.image_dict), len(SketchySet.pairs))

#print(os.listdir(sketchpath))
#print(os.listdir(imagepath))
print("Dataset length: ", len(SketchySet))
dataloader=DataLoader(SketchySet)
#print(len(SketchySet))
i=random.randint(0,len(SketchySet)-1)
sk,im = SketchySet[i]

#sk = sk.permute(1, 2, 0)
#im = im.permute(1, 2, 0)

# Convert the tensors back to numpy arrays
#sk = sk.numpy()
#im = im.numpy()

sk=np.array(sk)#.transpose((1,2,0))
im=np.array(im)#.transpose((1,2,0))
#print(im)
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(sk)
ax[0].set_title('Sketch')
ax[0].axis('off')

ax[1].imshow(im)
ax[1].set_title('image')
ax[1].axis('off')

plt.show()