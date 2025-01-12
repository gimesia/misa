# %%
import os
import nibabel as nib
import numpy as np

REGISTRATIONS_BASE_DIR = r'C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MISA\labs\labfinalboss\multiatlas\registrations'

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
int_atlas_filename = 'intensity_atlas.nii.gz'
prob_atlas_filename = 'prob_atlas.nii.gz'


def get_prob_atlases():
    paths = collect_paths_with_filename(REGISTRATIONS_BASE_DIR, prob_atlas_filename)
    ims = []
    for path in paths:
        ims.append(nib.load(path).get_fdata())
    return np.array(ims)

def get_int_atlases():
    paths = collect_paths_with_filename(REGISTRATIONS_BASE_DIR, int_atlas_filename)
    ims = []
    for path in paths:
        ims.append(nib.load(path).get_fdata())
    return np.array(ims)


probs = get_prob_atlases()
ints = get_int_atlases()
print(probs.shape)
print(ints.shape)

# %%
import numpy as np
from scipy.signal import correlate

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

def find_best_match(input_img, img_list):
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