# %%
import subprocess
import os
import shutil

import numpy as np
import nibabel as nib

from labfinalboss.multiatlas.utils import get_file_paths, REGISTRATION_DIR, DATASET_DIR, run_elastix_par0010


def process_images(fixed_image, image_paths):
    for moving_image in image_paths:
        run_elastix_par0010(fixed_image, moving_image)

# %%
file_paths = get_file_paths(DATASET_DIR)
print(file_paths)
file_paths["Training_Set"]["labels"]

# %%
fixed_image_path = file_paths["Validation_Set"]["img"][1] # IBSR_12.nii.gz
run_elastix_par0010(fixed_image_path, file_paths["Training_Set"]["img"][0])

# %%
fixed_image_path = file_paths["Validation_Set"]["img"][1]  # IBSR_12.nii.gz
training_images = file_paths["Training_Set"]["img"]
#process_images(fixed_image_path, training_images)
training_images

# %%
for fixed_image in file_paths["Training_Set"]["img"]:
    print(f"REGISTERING TRAINING SET ONTO: {fixed_image}")
    process_images(fixed_image, training_images)
    print("_________________________________________________________")