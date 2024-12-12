# %%
import subprocess
import os
import shutil

import numpy as np
import nibabel as nib


def get_file_paths(base_dir):
    data = {
        'Test_Set': {'img': [], 'labels': []},
        'Training_Set': {'img': [], 'labels': []},
        'Validation_Set': {'img': [], 'labels': []}
    }

    for subset in data.keys():
        subset_dir = os.path.join(base_dir, subset)
        for patient_dir in os.listdir(subset_dir):

            patient_path = os.path.join(subset_dir, patient_dir)
            if os.path.isdir(patient_path):
                img_path = os.path.join(patient_path, f'{patient_dir}.nii.gz')
                if os.path.exists(img_path): data[subset]['img'].append(img_path)
                labels_path = os.path.join(patient_path, f'{patient_dir}_seg.nii.gz')
                if os.path.exists(labels_path): data[subset]['labels'].append(labels_path)
    return data

def minmax_normalize(image):
    """Normalize the image to the range [0, 255] and convert to uint8."""
    image_min = np.min(image)
    image_max = np.max(image)
    normalized_image = (image - image_min) / (image_max - image_min) * 255
    return normalized_image.astype(np.uint8)

def preprocess_image(image_path, output_path):
    """Read, normalize, and save the NIfTI image."""
    img = nib.load(image_path)
    img_data = img.get_fdata()
    normalized_data = minmax_normalize(img_data)
    normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
    nib.save(normalized_img, output_path)

def modify_transform_parameters(file_path):
    """Modify the resample interpolator line in the TransformParameters file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if line.startswith('(ResampleInterpolator'):
                file.write('(ResampleInterpolator "FinalNearestNeighborInterpolator")\n')
            else:
                file.write(line)


def run_elastix_par0010(fixed_path, moving_path):
    # Path to the elastix & transformix executable
    elastix_exe = r"C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MIRA\lab\lab2\elastix-5.0.0-win64\elastix.exe"
    transformix_exe = r"C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MIRA\lab\lab2\elastix-5.0.0-win64\transformix.exe"
    # Path to the affine parameter file
    param_affine = r"C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MIRA\lab\lab2\elastix_model_zoo\models\Par0010\Par0010affine.txt"
    # Path to the bspline parameter file
    param_bspline = r"C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MIRA\lab\lab2\elastix_model_zoo\models\Par0010\Par0010bspline.txt"

    out_dir = moving_path.replace(r"dataset", rf"multiatlas\registered").replace(".nii.gz", "")  # Output directory

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    command = [
        elastix_exe,
        "-f", fixed_path,
        "-m", moving_path,
        "-p", param_affine,
        "-p", param_bspline,
        "-out", out_dir,
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())

    stderr = process.communicate()[1]
    if process.returncode != 0:
        print(f"Error running elastix: {stderr}")
    else:
        print("Elastix completed successfully.")



    # Transform the label image using transformix
    label_image = moving_path.replace(".nii.gz", "_seg.nii.gz")
    transform_param_file = os.path.join(out_dir, "TransformParameters.1.txt")
    modify_transform_parameters(transform_param_file)

    transform_command = [
        transformix_exe,
        "-in", label_image,
        "-out", out_dir,
        "-tp", transform_param_file,
    ]

    transform_process = subprocess.Popen(transform_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        output = transform_process.stdout.readline()
        if output == "" and transform_process.poll() is not None:
            break
        if output:
            print(output.strip())

    transform_stderr = transform_process.communicate()[1]
    if transform_process.returncode != 0:
        print(f"Error running transformix: {transform_stderr}")
    else:
        print("Transformix completed successfully.")

    # Delete all other files in the output directory except the specified ones
    keep_files = {"result.0.nii.gz", "result.1.nii.gz", "result.nii.gz",
                  "TransformParameters.0.txt", "TransformParameters.1.txt"}
    for filename in os.listdir(out_dir):
        if filename not in keep_files:
            file_path = os.path.join(out_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

# %%
base_dir = r'labfinalboss\dataset'
file_paths = get_file_paths(base_dir)
print(file_paths)

# %%
file_paths["Training_Set"]["labels"]

# %%
fixed_image = file_paths["Validation_Set"]["img"][1] # IBSR_12.nii.gz

run_elastix_par0010(fixed_image, file_paths["Training_Set"]["img"][0])

# %%
# run_elastix_par0010(fixed_image, file_paths["Training_Set"]["img"][0])

# %%
def process_images(fixed_image, image_paths):
    for moving_image in image_paths:
        if moving_image == fixed_image:
            # Copy the fixed image to the output directory
            out_dir = moving_image.replace(r"dataset", rf"multiatlas\registered").replace(".nii.gz", "")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            shutil.copy(fixed_image, os.path.join(out_dir, os.path.basename(fixed_image)))
        else:
            # Run elastix for other images
            run_elastix_par0010(fixed_image, moving_image)

# Example usage
fixed_image = file_paths["Validation_Set"]["img"][1]  # IBSR_12.nii.gz
all_images = file_paths["Training_Set"]["img"]
process_images(fixed_image, all_images)