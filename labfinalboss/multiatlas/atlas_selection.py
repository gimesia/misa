# %%
import os

import nibabel as nib
import numpy as np
from scipy.signal import correlate

from labfinalboss.multiatlas.em_atlas_segmentation import em_atlas_segmentation, atlas_seg
from labfinalboss.multiatlas.utils import collect_paths_with_filename, get_dataset_file_paths, REGISTRATION_DIR, \
    get_prob_atlases, get_int_atlases, read_nii, run_elastix_par0010, modify_transform_parameters, dice_score_per_class

probs, prob_paths = get_prob_atlases()
ints, int_paths = get_int_atlases()
print(probs.shape)
print(ints.shape)

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



# %
# run_elastix_on_all_atlases(dic["Validation_Set"]["img"][3])

# %
# for i in dic["Validation_Set"]["img"]:
#    run_elastix_on_all_atlases(i)

# %%
def normalized_cross_correlation(image1: np.ndarray, image2: np.ndarray) -> float:
    if image1.shape != image2.shape:
        raise ValueError(f"Input images must have the same shape. {image1.shape} != {image2.shape}")

    # Flatten the images for easier computation
    image1_flat = image1.ravel()
    image2_flat = image2.ravel()

    # Compute the means of the images
    mean1 = np.mean(image1_flat)
    mean2 = np.mean(image2_flat)

    # Subtract the means
    image1_zero_mean = image1_flat - mean1
    image2_zero_mean = image2_flat - mean2

    # Compute the numerator (dot product of zero-mean images)
    numerator = np.sum(image1_zero_mean * image2_zero_mean)

    # Compute the denominators (norms of zero-mean images)
    denominator = np.sqrt(np.sum(image1_zero_mean ** 2) * np.sum(image2_zero_mean ** 2))

    if denominator == 0:
        raise ValueError("Denominator is zero. This may occur if one or both images are constant.")

    # Compute NCC
    ncc = numerator / denominator

    return ncc

# %%
def predict_with_atlas(input_image_path):
    input_image = nib.load(input_image_path).get_fdata()
    input_image_name = os.path.split(input_image_path)[-1].replace(".nii.gz", "")

    directory = fr"{REGISTRATION_DIR}\atlas\fixed_{input_image_name}"
    prob_atlas_paths = collect_paths_with_filename(directory,"transformed_probability_map.nii.gz")
    int_atlas_paths = collect_paths_with_filename(directory,"result.1.nii.gz")

    intensity_atlases = np.stack([nib.load(f).get_fdata() for f in int_atlas_paths])
    # print(int_atlas_paths)
    # print(intensity_atlases.shape)
    ncc_s = [normalized_cross_correlation(np.squeeze(input_image), np.squeeze(im)) for im in intensity_atlases]
    # print(ncc_s)
    highest_ncc_index = np.argmax(ncc_s)

    # Read the probability atlas at the highest NCC index
    highest_prob_atlas_path = prob_atlas_paths[highest_ncc_index]
    highest_prob_atlas = nib.load(highest_prob_atlas_path).get_fdata()
    # print(highest_prob_atlas)
    print(f"Predicted with atlas {highest_ncc_index} with NCC {ncc_s[highest_ncc_index]}")
    #segment, _ = em_atlas_segmentation(input_image, highest_prob_atlas, num_classes=4, max_iters=150, tol=1e-3)
    segment = np.argmax(highest_prob_atlas, axis=-1)
    nib.save(nib.Nifti1Image(
        segment.astype(np.int32), nib.load(input_image_path).affine),
        fr"{directory}\EM-ATLAS-{input_image_name}_atlas{highest_ncc_index+1}.nii.gz")
    return segment

dices = []

for i in dic["Validation_Set"]["img"]:
    seg = predict_with_atlas(i)
    print(f"Segmentation for case {i} done")

    num_classes = np.unique(seg).size
    dice_scores = []
    gt = np.squeeze(nib.load(i.replace(".nii","_seg.nii")).get_fdata())

    for c in range(1,  num_classes):  # assuming labels start from 1 for tissues
        pred_class = (seg == c).astype(np.float32)
        gt_class = (gt == c).astype(np.float32)

        intersection = np.sum(pred_class * gt_class)
        union = np.sum(pred_class) + np.sum(gt_class)

        if union == 0:
            dice = 1.0  # If both the prediction and ground truth are empty, the Dice score is 1
        else:
            dice = 2.0 * intersection / union

        dice_scores.append((dice))

    print(f"Dices for case {i}: {dice_scores}, mean: {np.mean(dice_scores)}")
    dices.append(np.array(dice_scores))


print("Mean dice: " + np.mean(dices).__str__())
print("Mean dice: " + np.mean(dices, axis=1).__str__())
print("Mean dice: " + np.mean(dices, axis=2).__str__())
print("Mean dice: " + np.mean(dices, axis=0).__str__())