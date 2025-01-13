# %%
import os

import nibabel as nib
import numpy as np
from scipy.signal import correlate

from labfinalboss.multiatlas.utils import collect_paths_with_filename, get_dataset_file_paths, REGISTRATION_DIR, \
    get_prob_atlases, get_int_atlases, read_nii, run_elastix_par0010, modify_transform_parameters

probs, prob_paths = get_prob_atlases()
ints, int_paths = get_int_atlases()
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

def find_best_match_ncc(input_img, img_list, verbose=False):
    """
    Find the index of the image in img_list that has the highest normalized cross-correlation with input_img.

    :param input_img: The input 3D image.
    :param img_list: List or array of 3D images.
    :return: Index of the image with the highest normalized cross-correlation.
    """
    best_index = -1
    highest_ncc = -1

    for i, img in enumerate(img_list):
        if verbose:
            print(f"Comparing input image to image {i}")
        ncc = normalized_cross_correlation(np.squeeze(input_img), img)
        if ncc > highest_ncc:
            highest_ncc = ncc
            best_index = i

    return best_index

# %%
dic = get_dataset_file_paths()
dic

# %%
def run_elastix_on_all_atlases(input_image_path, override=False, transform=True):
    """
    Run elastix registration on all intensity atlases using the input image as the moving image.

    :param input_image_path: Path to the input image to be registered.
    :param override: Boolean indicating if existing registrations should be overridden.
    :param transform: Boolean indicating if the transformation part should be executed.
    """
    intensity_atlases, atlas_paths = get_int_atlases()

    for moving_image_path in atlas_paths:
        print(f"Running elastix for\n"+
              f"\tfixed image: {input_image_path}\n"
              f"\nmoving image: {moving_image_path}")
        run_elastix_par0010(input_image_path, moving_image_path, atlas=True, override=override)



# %%
run_elastix_on_all_atlases(dic["Validation_Set"]["img"][3])