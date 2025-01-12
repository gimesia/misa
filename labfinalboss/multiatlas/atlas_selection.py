# %%
import os
from pyclbr import readmodule

import nibabel as nib
import numpy as np
from scipy.signal import correlate

from labfinalboss.multiatlas.utils import collect_paths_with_filename, get_dataset_file_paths, REGISTRATION_DIR, \
    get_prob_atlases, get_int_atlases, read_nii

probs = get_prob_atlases()
ints = get_int_atlases()
print(probs.shape)
print(ints.shape)

# %%
def normalized_cross_correlation(img1, img2):
    """
    Compute the normalized cross-correlation between two 3D images.

    :param img1: First 3D image.
    :param img2: Second 3D image.
    :return: Normalized cross-correlation value.
    """
    img1_mean = np.mean(img1)
    img2_mean = np.mean(img2)
    img1_std = np.std(img1)
    img2_std = np.std(img2)

    numerator = np.sum((img1 - img1_mean) * (img2 - img2_mean))
    denominator = np.prod(img1.shape) * img1_std * img2_std

    return numerator / denominator

def find_best_match_ncc(input_img, img_list):
    """
    Find the index of the image in img_list that has the highest normalized cross-correlation with input_img.

    :param input_img: The input 3D image.
    :param img_list: List or array of 3D images.
    :return: Index of the image with the highest normalized cross-correlation.
    """
    best_index = -1
    highest_ncc = -1

    for i, img in enumerate(img_list):
        ncc = normalized_cross_correlation(input_img, img)
        if ncc > highest_ncc:
            highest_ncc = ncc
            best_index = i

    return best_index

# %%
dic = get_dataset_file_paths()
dic


# %%
def get_atlas_register(fixed_img_path):
    img = read_nii(fixed_img_path)

    prob_index = find_best_match_ncc(img, probs)
    print(f"Best probability atlas: {prob_index}")

