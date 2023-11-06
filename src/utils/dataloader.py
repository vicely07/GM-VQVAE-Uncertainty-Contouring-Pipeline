"""
The following is an import of PyTorch libraries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import numpy as np
#import nibabel as ni
import os, shutil
import time
import random
import pandas as pd
import numpy as np
import os
import cv2
import numpy as np
import os
import cv2
from scipy import ndimage
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def crop_around_centroid(array, dim1):
  i, j = ndimage.center_of_mass(array)
  i, j = int(i), int(j)
  w = int(dim1/2)
  imin = max(0,i-w)
  imax = min(array.shape[0],i+w+1)
  jmin = max(0,j-w)
  jmax = min(array.shape[1],j+w+1)
  crop =  array[imin:imax,jmin:jmax]
  return crop

def standard_resize2d(image, dim):
  resize_x,  resize_y = dim[0], dim[1]
  img_sm = cv2.resize(image, (resize_x, resize_y), interpolation=cv2.INTER_CUBIC)

  return img_sm


def split_train_test(dir, ratio_test=0.15):
    if not os.path.exists(os.path.join(dir, "train")): os.mkdir(os.path.join(dir, "train"))
    if not os.path.exists(os.path.join(dir, "test")): os.mkdir(os.path.join(dir, "test"))

    images_list = [i for i in os.listdir(dir) if i.endswith(".nii")]

    random.shuffle(images_list)
    threshold = int(len(images_list)*ratio_test)
    train_list = images_list[:-threshold]
    test_list = images_list[-threshold:]

    for i in train_list:
        shutil.move(os.path.join(dir, i), os.path.join(dir, "train", i))
    for i in test_list:
        shutil.move(os.path.join(dir, i), os.path.join(dir, "test", i))

def save_data_to_csv(dir, z):
    pd.DataFrame(z).to_csv(dir, header=None, index=False)

def postprocess_mask(mask, s):
  mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
  mask = np.where(mask >= 0.3, 1.0, 0.0)
  if s <= 15:
    #mask_blur = gaussian_filter(mask, sigma=7)
  #elif s > 10 and s <= 15:
    mask_blur = gaussian_filter(mask, sigma=5)
  else:
    mask_blur = gaussian_filter(mask, sigma=3)

  return mask_blur

def load_data_images(path, batch_size):
    filenames = [i for i in os.listdir(path) if i.endswith(".npy")] #and i.startswith("norm_023_S_0030")
    random.shuffle(filenames)
    n = 0
    while n < len(filenames):
        batch_image = []
        for i in range(n, n + batch_size):
            #print(filenames[i])
            if i >= len(filenames):
                ##n = i
                break

            image = np.load(os.path.join(path, filenames[i]), allow_pickle=True)#[1, ...]
            image = np.where(image >= 1e-3, image, 0.0)
            image = crop_around_centroid(image, dim1=240)
            image = np.pad(image, ((1,0), (1,0)), "constant", constant_values=0)
            dim = (256,256)
            image = torch.Tensor(standard_resize2d(image, dim))
            #image = random_rotate_transforms(image)
            image = torch.reshape(image, (1,1, 256, 256))
            image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            image = torch.where(image >= 0.1, image, 0.0)
            batch_image.append(image)

        n += batch_size
        batch_image = torch.cat(batch_image, axis=0)
        yield batch_image