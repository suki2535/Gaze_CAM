# # Install PyTorch with CUDA
#!pip install torch torchvision torchaudio

# # Install additional dependencies
#!pip install datasets matplotlib pandas pillow timm torcheval torchtnt==0.2.0 tqdm

# # Install utility packages
#!pip install cjm_pandas_utils cjm_pil_utils cjm_pytorch_utils cjm_torchvision_tfms

#!ls /content/drive/MyDrive/dataset/chest_xray

#from google.colab import drive
#drive.mount('/content/drive')

# Import Python Standard Library dependencies
from copy import copy
import datetime
from glob import glob
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import urllib.request

#import cv2

# Import utility functions
from cjm_pandas_utils.core import markdown_to_pandas
from cjm_pil_utils.core import resize_img, get_img_files
from cjm_psl_utils.core import download_file, file_extract
from cjm_pytorch_utils.core import set_seed, pil_to_tensor, tensor_to_pil, get_torch_device, denorm_img_tensor
from cjm_torchvision_tfms.core import ResizeMax, PadSquare

# Import matplotlib for creating plots
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

# Import pandas module for data manipulation
import pandas as pd

# Do not truncate the contents of cells and display all rows and columns
pd.set_option('max_colwidth', None, 'display.max_rows', None, 'display.max_columns', None)

# Import PIL for image manipulation
from PIL import Image

# Import timm library
import timm
import cv2
# Import PyTorch dependencies
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torchcam.methods import CAM
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2  as transforms
from torchvision.transforms.v2 import functional as TF

from torchtnt.utils import get_module_summary
from torcheval.metrics import MulticlassAccuracy

# Import tqdm for progress bar
from tqdm.auto import tqdm

"""## Setting Up the Project

### Setting a Random Number Seed
"""
#from IPython.display import display

#from cam1 import gen_cam

# Set the seed for generating random numbers in PyTorch, NumPy, and Python's random module.
seed = 1234
set_seed(seed)

"""### Setting the Device and Data Type"""

device = get_torch_device()
dtype = torch.float32
device, dtype



"""### Get Image Folders"""


#dataset_path = Path('D:/cxraytimdata/dataset/train')
#checkpoint_dir = Path('D:/cxraytimdata')
#gaze_path=Path('D:/cxraytimdata/gaze/train')
dataset_path = Path('D:/cxr_timm/dataset/train')
gaze_path= Path('D:/cxr_timm/gaze/train')
checkpoint_dir = Path('D:/cxr_timm')

img_folder_paths = [folder for folder in dataset_path.iterdir() if folder.is_dir()]
# Display the names of the folders using a Pandas DataFrame
pd.DataFrame({"Image Folder": [folder.name for folder in img_folder_paths]})

"""### Get Image File Paths"""

# Get a list of all image file paths from the image folders

class_file_paths = [get_img_files(folder) for folder in img_folder_paths]
# Get alBSl image files in the 'img_dir' directory
img_paths = [
    file
    for folder in class_file_paths # Iterate through each image folder
    for file in folder # Get a list of image files in each image folder
]

# Print the number of image files
print(f"Number of Images: {len(img_paths)}")

# Display the first five entries using a Pandas DataFrame
pd.DataFrame(img_paths).head()



"""### Inspecting the Class Distribution

#### Get image classes
"""

# Get the number of samples for each image class
class_counts_dict = {folder[0].parent.name:len(folder) for folder in class_file_paths}

# Get a list of unique labels
class_names = list(class_counts_dict.keys())

# Display the labels and the corresponding number of samples using a Pandas DataFrame
class_counts = pd.DataFrame.from_dict({'Count':class_counts_dict})
print(class_counts, class_names)

"""#### Visualize the class distribution"""

# Plot the distribution
class_counts.plot(kind='bar')
plt.title('Class distribution')
plt.ylabel('Count')
plt.xlabel('Classes')
plt.xticks(range(len(class_counts.index)), class_names)  # Set the x-axis tick labels
plt.xticks(rotation=75)  # Rotate x-axis labels
plt.gca().legend().set_visible(False)
#plt.show()

"""### Visualizing Sample Images"""

# Create a list to store the first image found for each class
sample_image_paths = [folder[0] for folder in class_file_paths]
sample_labels = [path.parent.stem for path in sample_image_paths]

# Calculate the number of rows and columns
grid_size = math.floor(math.sqrt(len(sample_image_paths)))
n_rows = grid_size+(1 if grid_size**2 < len(sample_image_paths) else 0)
n_cols = grid_size

# Create a figure for the grid
fig, axs = plt.subplots(n_rows, n_cols, figsize=(12,12))

for i, ax in enumerate(axs.flatten()):
    # If we have an image for this subplot
    if i < len(sample_image_paths) and sample_image_paths[i]:
        # Add the image to the subplot
        ax.imshow(np.array(Image.open(sample_image_paths[i])))
        # Set the title to the corresponding class name
        ax.set_title(sample_labels[i])
        # Remove the axis
        ax.axis('off')
    else:
        # If no image, hide the subplot
        ax.axis('off')

# Display the grid
plt.tight_layout()
#plt.show()



"""## Selecting a Model

### Exploring Available Models
"""

#RESTNET 18
#timmodels=timm.list_models('resnet18*', pretrained=True)
#pd.DataFrame(timmodels)
#print(timmodels)

#RESTNET 50
timmodels=timm.list_models('resnet50*', pretrained=True)
pd.DataFrame(timmodels)
print(timmodels)

"""### Inspecting the Model Configuration"""

# Import the resnet module
from timm.models import resnet as model_family

# RESTNET 18 Define the base model variant to use
#base_model = 'resnet18d'
#version = "ra2_in1k"

# RESTNET 50 Define the base model variant to use
print("RESTNET 50")
base_model = 'resnet50d'
version = "ra2_in1k"

# Get the default configuration of the chosen model
model_cfg = model_family.default_cfgs[base_model].default.to_dict()

# Show the default configuration values
pd.DataFrame.from_dict(model_cfg, orient='index')

"""### Retrieving Normalization Statistics"""

# Retrieve normalization statistics (mean and std) specific to the pretrained model
mean, std = model_cfg['mean'], model_cfg['std']
norm_stats = (mean, std)
norm_stats

"""### Loading the Model"""

# Create a pretrained ResNet model with the number of output classes equal to the number of class names
# 'timm.create_model' function automatically downloads and initializes the pretrained weights
model = timm.create_model(f'{base_model}.{version}', pretrained=True, num_classes=len(class_names))

# Set the device and data type for the model
model = model.to(device=device, dtype=dtype)

# Add attributes to store the device and model name for later reference
model.device = device
model.name = f'{base_model}.{version}'

"""### Summarizing the Model"""

# Define the input to the model
test_inp = torch.randn(1, 3, 256, 256).to(device)

# Get a summary of the model as a Pandas DataFrame
summary_df = markdown_to_pandas(f"{get_module_summary(model, [test_inp])}")

# Filter the summary to only contain Conv2d layers and the model
summary_df = summary_df[(summary_df.index == 0) | (summary_df['Type'] == 'Conv2d')]

# Remove the column "Contains Uninitialized Parameters?"
summary_df.drop('Contains Uninitialized Parameters?', axis=1)

"""## Preparing the Data

### Training-Validation Split
"""

# Shuffle the image paths
random.shuffle(img_paths)

# Define the percentage of the images that should be used for training
train_pct = 0.9
val_pct = 0.1

# Calculate the index at which to split the subset of image paths into training and validation sets
train_split = int(len(img_paths)*train_pct)
val_split = int(len(img_paths)*(train_pct+val_pct)) + 1

# Split the subset of image paths into training and validation sets
train_paths = img_paths[:train_split]
val_paths = img_paths[train_split:]

# Print the number of images in the training and validation sets
pd.Series({
    "Training Samples:": len(train_paths),
    "Validation Samples:": len(val_paths)
}).to_frame().style.hide(axis='columns')

"""### Data Augmentation

#### Set training image size
"""

train_sz = 224

"""#### Initialize image transforms"""

# Set the fill color for padding images
fill = (0,0,0)

# Create a `ResizeMax` object
resize_max = ResizeMax(max_sz=train_sz)

# Create a `PadSquare` object
pad_square = PadSquare(shift=True, fill=fill)

# # Create a TrivialAugmentWide object
#trivial_aug = transforms.TrivialAugmentWide(fill=fill)



"""#### Test the transforms"""

sample_img = Image.open(img_paths[11]).convert('RGB')
sample_img
print("Display image")
#display(sample_img)

# Augment the image
#augmented_img = trivial_aug(sample_img)

# Resize the image
#resized_img = resize_max(augmented_img)

# Pad the image
#padded_img = pad_square(resized_img)

# Ensure the padded image is the target size
#resize = transforms.Resize([train_sz] * 2, antialias=True)
#resized_padded_img = resize(padded_img)

# Display the annotated image
#display(resized_padded_img)

'''pd.Series({
    "Source Image:": sample_img.size,
    "Resized Image:": resized_img.size,
    "Padded Image:": padded_img.size,
    "Resized Padded Image:": resized_padded_img.size,
}).to_frame().style.hide(axis='columns')'''



"""### Training Dataset Class"""

class ImageDataset(Dataset):
    """
    A PyTorch Dataset class for handling images.

    This class extends PyTorch's Dataset and is designed to work with image data.
    It supports loading images, and applying transformations.

    Attributes:
        img_paths (list): List of image file paths.
        class_to_idx (dict): Dictionary mapping class names to class indices.
        transforms (callable, optional): Transformations to be applied to the images.
    """

    def __init__(self, img_paths, class_to_idx, transforms=None, gaze_transforms = None):
        """
        Initializes the ImageDataset with image keys and other relevant information.

        Args:
            img_paths (list): List of image file paths.
            class_to_idx (dict): Dictionary mapping class names to class indices.
            transforms (callable, optional): Transformations to be applied to the images.
        """
        super(Dataset, self).__init__()

        self._img_paths = img_paths
        self._class_to_idx = class_to_idx
        self._transforms = transforms
        self._gaze_tfm = gaze_transforms

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self._img_paths)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        img_path = self._img_paths[index]
        image, label = self._load_image(img_path)
        g_image = self._load_gaze(label, img_path)
        # Applying transformations if specified
        if self._transforms:
            image = self._transforms(image)
        if self._gaze_tfm:
            g_image = self._gaze_tfm(g_image)
        return image,g_image, label



    def _load_gaze(self, label,img_path):
        dicomid=(os.path.basename(img_path).split('.')[0])
        g_path = str(gaze_path)
        g_path = g_path + "\\" + class_names[label] + "\\" + dicomid + "_gazemap"+ ".png"
        if (os.path.isfile(g_path)):
            gazeimage = Image.open(g_path).convert('RGB')
            return gazeimage
        else:
            print(g_path)
    def _load_image(self, img_path):
        """
        Loads an image from the provided image path.

        Args:
            img_path (string): Image path.
            Returns:
        tuple: A tuple containing the loaded image and its corresponding target data.
        """
        # Load the image from the file path
        image = Image.open(img_path).convert('RGB')
#        print("load_image")
        return image, self._class_to_idx[img_path.parent.name]

"""### Image Transforms"""

# Compose transforms to resize and pad input images
resize_pad_tfm = transforms.Compose([
    resize_max,
    pad_square,
    transforms.Resize([train_sz] * 2, antialias=True)
])

# Compose transforms to sanitize bounding boxes and normalize input data
final_tfms = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(*norm_stats),
])

# Define the transformations for training and validation datasets
# Note: Data augmentation is performed only on the training dataset
train_tfms = transforms.Compose([
    resize_pad_tfm,
    final_tfms
])
grayscale_transform = transforms.Grayscale(num_output_channels=1)
valid_tfms = transforms.Compose([resize_pad_tfm, final_tfms])
gaze_tfms  = transforms.Compose([
    resize_pad_tfm,
    transforms.ToImage(),
    grayscale_transform,
    transforms.ToDtype(torch.float32, scale=True),
])


"""### Initialize Datasets"""

# Create a mapping from class names to class indices
class_to_idx = {c: i for i, c in enumerate(class_names)}
print(class_to_idx)
# Instantiate the dataset using the defined transformations
train_dataset = ImageDataset(train_paths, class_to_idx, train_tfms, gaze_tfms)
valid_dataset = ImageDataset(val_paths, class_to_idx, valid_tfms, gaze_tfms)

# Print the number of samples in the training and validation datasets
pd.Series({
    'Training dataset size:': len(train_dataset),
    'Validation dataset size:': len(valid_dataset)}
).to_frame().style.hide(axis='columns')

"""### Inspect Samples

**Inspect training set sample**
"""
# Get the label for the first image in the training set
'''print(f"Label: {class_names[train_dataset[0][1]]}")

# Get the first image in the training set
TF.to_pil_image(denorm_img_tensor(train_dataset[0][0], *norm_stats))

"""**Inspect validation set sample**"""

# Get the label for the first image in the validation set
print(f"Label: {class_names[valid_dataset[0][1]]}")

# Get the first image in the validation set
TF.to_pil_image(denorm_img_tensor(valid_dataset[0][0], *norm_stats))
'''

"""### Training Batch Size"""

bs = 32

"""### Initialize DataLoaders"""

# Set the number of worker processes for loading data. This should be the number of CPUs available.
num_workers = multiprocessing.cpu_count()#//2

# Define parameters for DataLoader
data_loader_params = {
    'batch_size': bs,  # Batch size for data loading
#    'num_workers': num_workers,  # Number of subprocesses to use for data loading
#    'persistent_workers': True,  # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the worker dataset instances alive.
#    'pin_memory': 'cuda' in device,  # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using GPU.
#    'pin_memory_device': device if 'cuda' in device else '',  # Specifies the device where the data should be loaded. Commonly set to use the GPU.
}

# Create DataLoader for training data. Data is shuffled for every epoch.
train_dataloader = DataLoader(train_dataset, **data_loader_params, shuffle=True)

# Create DataLoader for validation data. Shuffling is not necessary for validation data.
valid_dataloader = DataLoader(valid_dataset, **data_loader_params)

# Print the number of batches in the training and validation DataLoaders
print(f'Number of batches in train DataLoader: {len(train_dataloader)}')
print(f'Number of batches in validation DataLoader: {len(valid_dataloader)}')

"""## Fine-tuning the Model

### Define the Training Loop
"""

# Function to run a single training/validation epoch
def run_epoch(model, dataloader, optimizer, metric, lr_scheduler, device, scaler, epoch_id, is_training):
    # Set model to training mode if 'is_training' is True, else set to evaluation mode
    #model.train() if is_training else model.eval()

    # Reset the performance metric
    metric.reset()
    # Initialize the average loss for the current epoch
    epoch_loss = 0
    # Initialize progress bar with total number of batches in the dataloader
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")
    # Iterate over data batches
    for batch_id, (inputs,gaze,label) in enumerate(dataloader):
        n = len(label)
        model.eval()
        cam_extractor = CAM(model, target_layer='layer4')
        outputs = model(inputs)
#        if batch_id > 15:
#            break
        act = []
        for i in range(n):
            activation_map = cam_extractor(label[i].item(), outputs[i].unsqueeze(0))
            act.append(activation_map[0][i])
        # Move inputs and targets to the specified device (e.g., GPU)

        inputs, label = inputs.to(device), label.to(device)

        # Enables gradient calculation if 'is_training' is True
#        with torch.set_grad_enabled(is_training):
#            # Automatic Mixed Precision (AMP) context manager for improved performance
#            with autocast(torch.device(device).type):
#                outputs = model(inputs) # Forward pass
#                loss = torch.nn.functional.cross_entropy(outputs, targets) # Compute loss
        #print(inputs.size())
        if is_training:
            model.train()
            outputs = model(inputs)# Forward pass
        #print(outputs.size())

        batch_loss = torch.tensor(0.0)
        #print(gaze.size())
        for i in range(n):
            # print(label[i],outputs[i])
            map = act[i].unsqueeze(0).unsqueeze(0)
            resized_act = torch.nn.functional.interpolate(map, size=(224,224), mode='bicubic', align_corners=False)
            #(gaze[i].size())
            mask = resized_act.squeeze(0).squeeze(0).flatten()
            mask.requires_grad_()
            #print(mask.size())
            mseloss = torch.nn.functional.mse_loss(mask, gaze[i].flatten(), reduction="mean")
            #print(mseloss)
            batch_loss += mseloss
        loss0 = batch_loss/n
        #loss1 = torch.nn.functional.cross_entropy(outputs, targets) # Compute loss
        # Update the performance metric
        # mean_loss = torch.nn.functional.mse_loss(loss,  reduction="mean")
        loss1 = torch.nn.functional.cross_entropy(outputs, label)
        loss = loss1 + 0.5*loss0
        #loss = loss0
        metric.update(outputs.detach().cpu(), label.detach().cpu())
        #If in training mode
        if is_training:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()


        loss_item = loss.item()
        epoch_loss += loss_item
        # Update progress bar
        progress_bar.set_postfix(accuracy=metric.compute().item(),
                                 loss=loss_item,
                                 avg_loss=epoch_loss/(batch_id+1),
                                  lr=lr_scheduler.get_last_lr()[0] if is_training else "")
        progress_bar.update()

        # If loss is NaN or infinity, stop training
        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    progress_bar.close()
    return epoch_loss / (batch_id + 1)

# Main training loop
def train_loop(model, train_dataloader, valid_dataloader, optimizer, metric, lr_scheduler, device, epochs, checkpoint_path, use_scaler=False):
    # Initialize a gradient scaler for mixed-precision training if the device is a CUDA GPU
    scaler = GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')

    # Iterate over each epoch
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Run training epoch and compute training loss
        train_loss = run_epoch(model, train_dataloader, optimizer, metric, lr_scheduler, device, scaler, epoch, is_training=True)
        # Run validation epoch and compute validation loss
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, metric, None, device, scaler, epoch, is_training=False)

        # If current validation loss is lower than the best one so far, save model and update best loss
        if valid_loss < best_loss:
            best_loss = valid_loss
            metric_value = metric.compute().item()
            torch.save(model.state_dict(), checkpoint_path)

            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'metric_value': metric_value,
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            print(training_metadata)
            # Save best_loss and metric_value in a JSON file
            with open(Path(checkpoint_path.parent/'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)

    # If the device is a GPU, empty the cache
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()

"""### Set the Model Checkpoint Path"""

# Generate timestamp for the training session (Year-Month-Day_Hour_Minute_Second)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create the checkpoint directory if it does not already exist
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# The model checkpoint path
checkpoint_path = checkpoint_dir/f"{model.name}.pth"

print(checkpoint_path)

"""### Saving the Class Labels"""

# Save class labels
class_labels = {"classes": list(class_names)}

# Set file path
class_labels_path = checkpoint_dir/f"{model.name}-classes.json"

# Save class labels in JSON format
with open(class_labels_path, "w") as write_file:
    json.dump(class_labels, write_file)

print(class_labels_path)

"""### Configure the Training Parameters"""

# Learning rate for the model
lr = 5e-5

# Number of training epochs
epochs = 10

# AdamW optimizer; includes weight decay for regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = 0.001)

# Learning rate scheduler; adjusts the learning rate during training
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                   max_lr=lr,
                                                   total_steps=epochs*len(train_dataloader))

# Performance metric: Multiclass Accuracy
metric = MulticlassAccuracy()

"""### Train the Model"""

def train():
    train_loop(model=model,
           train_dataloader=train_dataloader,
           valid_dataloader=valid_dataloader,
           optimizer=optimizer,
           metric=metric,
           lr_scheduler=lr_scheduler,
           device=torch.device(device),
           epochs=epochs,
           checkpoint_path=checkpoint_path,
           use_scaler=True)

def predictxray(test_file):
    # Choose an item from the validation set
    print("Prediction:")
    print(test_file)

    # Open the test file
    test_img = Image.open(test_file).convert('RGB')

    # Set the minimum input dimension for inference
    input_img = resize_img(test_img, target_sz=train_sz, divisor=1)

    # Convert the image to a normalized tensor and move it to the device
    img_tensor = pil_to_tensor(input_img, *norm_stats).to(device=device)

    # Make a prediction with the model
    with torch.no_grad():
        pred = model(img_tensor)

    # Scale the model predictions to add up to 1
    pred_scores = torch.softmax(pred, dim=1)

    # Get the highest confidence score
    confidence_score = pred_scores.max()

    # Get the class index with the highest confidence score and convert it to the class name
    pred_class = class_names[torch.argmax(pred_scores)]

    # Display the image
#    display(test_img)

    print(f"Predicted Class: {pred_class}")

    # Print the prediction data as a Pandas DataFrame for easy formatting
    confidence_score_df = pd.DataFrame({
        'Confidence Score': {
            name: f'{score * 100:.2f}%' for name, score in zip(class_names, pred_scores.cpu().numpy()[0])
        }
    })
    print(confidence_score_df)

train()

'''or batchid in enumerate(train_dataloader):
    print("Load gaze date")
    break'''

#Load the pretrained model
#model.load_state_dict(torch.load('D:/cxraytimdata/resnet18d.ra2_in1k.pth', weights_only=True))
#model.load_state_dict(torch.load('D:/cxr_timm/resnet50d.ra2_ink1.pth', weights_only=True))
#model.eval()

#predictions
#for i in  range (0,20):
#    predictxray(val_paths[i])

# Generate CAM for an image
#imgpath = Path('D:/cxraytimdata/cam/dogcat.jpg')
#imgpath = Path('D:/cxraytimdata/cam/camtest.jpg')
#imgdpath = Path('D:/cxraytimdata/cam/dogcatcam.jpg')
#gen_cam(model,imgpath,imgdpath)



