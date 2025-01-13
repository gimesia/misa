# %%
import os
import nibabel as nib
import numpy as np

from labfinalboss.multiatlas.utils import REGISTRATION_DIR, collect_paths_with_filename

REGISTRATIONS_BASE_DIR = REGISTRATION_DIR

def create_intensity_atlas(registration_dir):
    fixed_name = os.path.split(registration_dir)[-1].replace("fixed_", "")
    paths = collect_paths_with_filename(registration_dir, 'result.1.nii.gz')
    extra_paths = collect_paths_with_filename(registration_dir, fixed_name)

    print(f"Creating intensity atlas for fixed {fixed_name}")
    paths = [*paths, *extra_paths]

    acc_img = None
    nii_img = None
    for path in paths:
        nii_img = nib.load(path)
        img = nii_img.get_fdata()

        if acc_img is None:
            acc_img = img

        acc_img += img

    acc_img /= len(paths)
    acc_img = nib.Nifti1Image(acc_img, nii_img.affine)

    # Save the resulting intensity atlas
    print(f"Saving intensity atlas to {registration_dir}")
    nib.save(acc_img, rf"{registration_dir}\intensity_atlas.nii.gz")

    return acc_img.get_fdata()

def create_probability_atlas(registration_dir):
    fixed_name = os.path.split(registration_dir)[-1].replace("fixed_", "")
    paths = collect_paths_with_filename(registration_dir, 'result.nii.gz')
    extra_paths = collect_paths_with_filename(registration_dir, fixed_name)

    print(f"Creating probability atlas for fixed {fixed_name}")
    paths = [*paths, *extra_paths]

    accumulated_img = None
    nii_img = None
    for path in paths:
        nii_img = nib.load(path)
        img = nii_img.get_fdata()

        if accumulated_img is None:
            accumulated_img = np.zeros((*img.shape, 4))
            # print(accumulated_img.shape)

        for i in range(0,4):
            class_label_mask_img = img == i
            # print(class_label_mask_img.shape)
            accumulated_img[:,:,:,i] += class_label_mask_img


    accumulated_img /= len(paths)
    accumulated_img = nib.Nifti1Image(accumulated_img, nii_img.affine)

    # Save the resulting intensity atlas
    print(f"Saving probability atlas to {registration_dir}")
    nib.save(accumulated_img, rf"{registration_dir}\prob_atlas.nii.gz")

    return accumulated_img.get_fdata()


# %% Example usage
intensity_atlas = create_intensity_atlas(f"{REGISTRATIONS_BASE_DIR}\\fixed_IBSR_12")
prob_atlas = create_probability_atlas(f"{REGISTRATIONS_BASE_DIR}\\fixed_IBSR_12")

# %% CREATING ATLAS FOR ALL REGISTRATIONS
for dirname in os.listdir(REGISTRATIONS_BASE_DIR):
    # print(dirname)
    dirname = os.path.join(REGISTRATIONS_BASE_DIR, dirname)
    intensity_atlas = create_intensity_atlas(dirname)
    prob_atlas = create_probability_atlas(dirname)



