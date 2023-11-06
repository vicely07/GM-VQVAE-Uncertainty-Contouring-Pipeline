import os
from dicom_utils import dcm2npy
import argparse
import nibabel as nib
import numpy as np

def get_conversion_args():
    parser = argparse.ArgumentParser(
            description="Use this script for data conversion "
                        ,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
            "--input_path",
            type=str,
            default="datasets/prostate_data.npy",
            help="Filepath to test data",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Filepath to output folder",
    )
    return parser.parse_args()

def dcm2nii(input_path, output_path):
  data, _, _, _, _ = dcm2npy(input_path)
  data_nii = nib.Nifti1Image(np.rot90(data, k = 2).transpose((2, 1, 0)), affine=np.eye(4))
  file_name = input_path.split("/")[-1]
  print(os.path.join(output_path, file_name+"_0000.nii.gz"))
  print(data_nii.shape)
  nib.save(data_nii, os.path.join(output_path, file_name+"_0000.nii.gz"))

def nii2npy(input_path, output_path):
  data = nib.load(input_path).get_data()
  file_name = input_path.split("/")[-1]
  np.save(os.path.join(output_path, file_name, data))

# Load input args
args = get_conversion_args()
input_path, output_path = args.input_path, args.output_path
subject_list = [x[1] for x in os.walk(input_path)][0]
print(subject_list)
for subject in subject_list:
    dcm2nii(os.path.join(input_path,subject), output_path)
    #except:
    #    pass
