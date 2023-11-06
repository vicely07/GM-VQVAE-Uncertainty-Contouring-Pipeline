import glob
import torch 
from gm_vqvae import VQVAE, Encoder, ResBlock, Quantize, Decoder
import numpy as np 
import pandas as pd 

def evaluate_step(path_data_list, path_model_list, model_name_list, threshold, label_list):
  mean_dsc_list = []
  mean_hd_list = []
  for i in range(len(path_model_list)):
    mean_dsc, mean_hd = evaluate_testset(path_data_list[i], path_model_list[i], threshold, label_list[i])
    mean_dsc_list.append(mean_dsc)
    mean_hd_list.append(mean_hd)

  metric_tbl  = pd.DataFrame(list(zip(model_name_list, mean_dsc_list, mean_hd_list)), columns =['Model', 'Surface DSC', "Hausdroff Distance"])
  return metric_tbl

def evaluate_testset(path_data, path_model, threshold, label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_model = torch.load(path_model, map_location=torch.device('cpu'))
    vae_model.to(device)
    dice_list = []
    hausdroff_list = []
    gt_list = glob.glob(path_data + "/*.npy")
    for gt in gt_list:
      if label == "all":
        gt = np.load(gt)
      else:
        gt = np.load(gt)[1, ...]
      gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
      gt = np.where(gt >= 0.1, 1.0, 0.0)
      # interfere phrase
      vae_model.eval()
      mask = pred_roi(gt, vae_model)
      # Mask
      #display_image3d(gt)
      mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
      mask = mask[1, ...]
      mask_t = np.where(mask >= threshold, 1.0, 0.0)
      #display_image3d(mask_t)
      sd = calculate_surface_dsc(mask_t.astype("bool"), gt.astype("bool"))
      print(sd)
      #mask_t = np.where(mask >= 3, 1.0, 0.0)
      hd = calculate_hausoff_dist(mask_t.astype("bool"), gt.astype("bool"))
      print(hd)
      dice_list.append(sd)
      hausdroff_list.append(hd)
    return np.mean(dice_list), np.mean(hausdroff_list)

def pred_roi(gt, vae_model):
    gt = np.where(gt >= 0.2, gt, 0.0)
    pred = np.zeros(gt.shape)
    with torch.no_grad():
      for i in range(gt.shape[0]):
        if np.sum(gt[i, ...]) > 0:
          slice = gt.copy()[i, ...]
          crop, imin, imax, jmin, jmax = crop_around_centroid(slice, dim1=200)
          x = load_data_images(crop)
          y, latent_loss = vae_model(x.to(device))
          y = y.detach().cpu().numpy()
          dim = crop.shape
          y = standard_resize2d(y[0, 0, ...], dim)
          pred[i, imin:imax, jmin:jmax] = y

    return gt, pred

def crop_around_centroid(array, dim1):
  i, j = ndimage.center_of_mass(array)
  i, j = int(i), int(j)
  w = int(dim1/2)
  imin = max(0,i-w)
  imax = min(array.shape[0],i+w+1)
  jmin = max(0,j-w)
  jmax = min(array.shape[1],j+w+1)
  crop =  array[imin:imax,jmin:jmax]

  return crop, imin, imax, jmin, jmax

def load_data_images(image):
  image = np.pad(image, ((1,0), (1,0)), "constant", constant_values=0)
  dim = (256,256)
  image = torch.Tensor(standard_resize2d(image, dim))
  image = torch.reshape(image, (1,1, 256, 256))
  image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
  return image

## Path data list
path_data = "../data/nnUnet-segmentation/Joint/"
path_data_prostate = "../data/nnUnet-segmentation/Prostate/"
path_data_rectum = "../data/nnUnet-segmentation/Rectum/"
path_data_bladder = "../data/nnUnet-segmentation/Bladder/"

## Model list
gm_vqvae = "../../checkpoints/gmvqvae/model_gmvqvae_best_joint.pt"
gm_vqvae_prostate = "../../checkpoints/gmvqvae/model_gmvqvae_best_prostate.pt"
gm_vqvae_rectum = "../../checkpoints/gmvqvae/model_gmvqvae_best_rectum.pt"
gm_vqvae_bladder = "../../checkpoints/gmvqvae/model_gmvqvae_best_bladder.pt"

threshold = 0.4
model_name_list = ["GM-VQVAE", "GM-VQVAE-Prostate", "GM-VQVAE-Rectum", "GM-VQVAE-Bladder"]
path_model_list = [gm_vqvae, gm_vqvae_prostate, gm_vqvae_rectum, gm_vqvae_bladder]
path_data_list = [path_data, path_data_prostate , path_data_rectum, path_data_bladder]
roi_list = ["all", "prostate", "rectum", "bladder"]

metric_table = evaluate_step(path_data_list, path_model_list, model_name_list, threshold, roi_list)
metric_table