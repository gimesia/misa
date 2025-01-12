# %%
import os
import nibabel as nib
import numpy as np


def collect_paths_with_filename(directory, filename):
    """
    Collect all paths in the given directory and its subdirectories that have the specified filename.

    :param directory: The base directory to search in.
    :param filename: The specific filename to look for.
    :return: A list of paths that match the specified filename.
    """
    matching_paths = []
    for root, _, files in os.walk(directory):
        if filename in files:
            matching_paths.append(os.path.join(root, filename))
    return matching_paths

# Example usage
directory = r'C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MISA\labs\labfinalboss\multiatlas\registrations\fixed_IBSR_12'
label_imgs = 'result.nii.gz'
registered_imgs = 'result.1.nii.gz'
paths = collect_paths_with_filename(directory, label_imgs)
print(paths)

# %%
def create_intensity_atlas(registration_dir):
    paths = collect_paths_with_filename(registration_dir, 'result.1.nii.gz')

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
    nib.save(acc_img, rf"{registration_dir}\intensity_atlas.nii.gz")

    return acc_img.get_fdata()

# %%
def create_probability_atlas(registration_dir):
    paths = collect_paths_with_filename(registration_dir, 'result.nii.gz')

    accumulated_img = None
    nii_img = None
    for path in paths:
        nii_img = nib.load(path)
        img = nii_img.get_fdata()

        if accumulated_img is None:
            accumulated_img = np.zeros((*img.shape, 4))
            print(accumulated_img.shape)

        for i in range(0,4):
            class_label_mask_img = img == i
            print(class_label_mask_img.shape)
            accumulated_img[:,:,:,i] += class_label_mask_img


    accumulated_img /= len(paths)
    accumulated_img = nib.Nifti1Image(accumulated_img, nii_img.affine)

    # Save the resulting intensity atlas
    nib.save(accumulated_img, rf"{registration_dir}\prob_atlas.nii.gz")

    return accumulated_img.get_fdata()


# Example usage
registration_dir = r'C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MISA\labs\labfinalboss\multiatlas\registrations\fixed_IBSR_12'
intensity_atlas = create_intensity_atlas(registration_dir)
prob_atlas = create_probability_atlas(registration_dir)
