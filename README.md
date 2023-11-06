## Abstract:
This study focuses on addressing uncertainties in contouring for radiation therapy planning, aiming to create a simulation algorithm that introduces controlled errors into contours. Using a dataset of CT scans from 159 prostate cancer patients, they employ a pipeline with a GM-VQVAE reconstruction model and size-shape alteration layers. For the task of ROI reconstruction, we found that the GM-VQVAE outperforms all other VAE-based models with Surface DSC (HD) of 0.964 (23.537), 0.934 (25.815), 0.972 (21.013), 0.945 (20.912) for prostate, rectum, bladder and joint ROI respectively. For the task of shape and size alteration, we found the shape parameter s ϵ [1, 10] and size parameter thresh ϵ [0.3, 0.7] constantly produce acceptable uncertainty contours in geometric evaluation. The proposed methodology generates realistic uncertainty contours, but further clinical review is needed to assess the clinical acceptability on the prostate and surrounding organs at risk.

## Methods:
For the task of uncertainty contouring, we will build a pipeline with GM-VQVAE generator in the center. Our deep learning approach focuses on the probability map from a segmentation algorithm. The probability map is then reconstructed using GM-VQVAE to extract the latent space. In brief, GM-VQVAE is a VAE with discrete latent space as a result from vector quantisation (VQ) and scaled attention from Gaussian Mixture (GM). The encoder architecture is composed of two strided convolutional layers with a stride of 2 and a window size of 4 × 4, succeeded by two residual blocks with a 3 × 3 configuration (implemented as ReLU activation, followed by a 3x3 convolution, another ReLU activation, and a 1x1 convolution), all containing 256 hidden units. Similarly, the decoder comprises two residual blocks followed by two transposed convolutions with a stride of 2 and a window size of 4 × 4.  The the number of clusters for GM is 64, number of discrete latent space for VQ is 1024, the number of residual blocks is 2, residual channels is 8 and convolutional channels is 128 We utilize the Adam optimizer with a learning rate of 1e-6 and evaluate the model's performance after 20,000-71,000 training steps depending on the organ, using a batch size of 32.

Fig 1 - Pipeline

In building the GM-VQVAE model, we introduced the concept of "GM as Weighted Scale". This means that the continuous latent space z from GM-based encoder is used as the weighted scale vector.

Fig 2 - Model

The latent space from the GM-VQVAE model is a good representation of the original probability map in a lower dimension and, because this space is Gaussian distributed, we can modify this space by adjusting the mean to control to control the shape. To extract uncertainty contours, we process the adjusted probability map through a thresholding layer to control the size. We have the flexibility to control the contour's size by varying the threshold levels on the Gaussian probability map. Given the multiple options for shaping (mean-level adjustment) and sizing (threshold adjustment), we systematically evaluate distance-based metrics to identify an optimal set of parameters that consistently yield acceptable uncertainty contours.

Fig 3

As in table 1, we found that the GM-VQVAE outperforms all other models with Surface DSC (HD) of 0.964 (23.537), 0.934 (25.815), 0.972 (21.013), 0.945 (20.912) for prostate, rectum, bladder and joint ROI respectively. 

 Table 1

# Exportation for Clinical Use:
## Install
pip install -r requirement.txt
git clone https://github.com/deepmind/surface-distance.git
pip install surface-distance/

## Export:
python export.py

## Training & Evaluation
## Dataset:
We used a dataset of deidentified CT scans extracted from MD Anderson Cancer clinical software (Ray Station), and the data is deidentified. Regarding the inclusion/exclusion criteria, we include subjects who have all 3 ROI of the prostate, bladder, and rectum presented in the CT scans. Thus, we exclude subjects who are labeled “prostate-fossa”, which means the subject has their prostate removed due to cancer. The data consists of 159 subjects. Due to the difference in length of the ROIs in the human body, the number of 2D slices are consequently split into 8,956 for prostates, 22,668 for rectum, 6,334 for bladder, and 22,668 for joint ROI. The training-validation-testing ratio is 80-10-10. In our training, we use a batch size of 32.


# Prepare the data:
## Extract nnUnet segmentation:
## 1. Process data for nnUnet:
cd utils
python dcm2nii.py --input_path ..data/input/dicom_ct/ --output_path ..data/input/nifti_formatted_ct/

## 2. nnUnet segmentation extraction:
cd nnUnet/nnUNet
pip install -e .
nnUNet_predict -i ../data/input/seg/nifti_formatted_ct/ -o ../data/input/seg/3d_masks -t Task201_Prostate -m 3d_fullres
nnUNet_predict -i ../data/input/seg/nifti_formatted_ct/ -o ../data/input/seg/3d_masks -t Task202_Bladder -m 3d_fullres
nnUNet_predict -i ../data/input/seg/nifti_formatted_ct/ -o ../data/input/seg/3d_masks -t Task203_Rectum -m 3d_fullres

## 3. Extract slices from 3d segmentation masks:
python extract_slices --input_path ../../data/seg/3d_masks/Prostate --output_mask ../../data/seg/2d_slices

## Organize slices into test

## Training:
python train.py

## Evaluation:
python eval.py