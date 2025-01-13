import os
import shutil
import subprocess

import nibabel as nib
import numpy as np

DATASET_DIR = r'C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MISA\labs\labfinalboss\dataset'
ATLAS_DIR = r'C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MISA\labs\labfinalboss\multiatlas'
REGISTRATION_DIR = r'C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MISA\labs\labfinalboss\multiatlas\registrations'

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

def run_elastix_par0010(fixed_path, moving_path, atlas=False, override=False):
    """
    Run elastix registration with the specified fixed and moving images.

    :param fixed_path: Path to the fixed image.
    :param moving_path: Path to the moving image. (If atlas is True, this should be the atlas)
    :param atlas: Boolean indicating if the registration is for an atlas.
    :param override: Boolean indicating if existing registrations should be overridden.
    """
    elastix_exe = r"C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MIRA\lab\lab2\elastix-5.0.0-win64\elastix.exe"
    transformix_exe = r"C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MIRA\lab\lab2\elastix-5.0.0-win64\transformix.exe"
    param_affine = r"C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MIRA\lab\lab2\elastix_model_zoo\models\Par0010\Par0010affine.txt"
    param_bspline = r"C:\Users\gimes\OneDrive\MAIA\3_UdG\classes\MIRA\lab\lab2\elastix_model_zoo\models\Par0010\Par0010bspline.txt"

    if atlas:
        moving_image_name = os.path.split(os.path.split(moving_path)[-2])[-1].__str__().replace("fixed", "atlas")
        fixed_image_name = os.path.split(fixed_path)[-1].replace(".nii.gz", "")
        out_dir = os.path.join(REGISTRATION_DIR, "atlas", f"fixed_{fixed_image_name}", moving_image_name)
    else:
        fixed_image_name = os.path.split(fixed_path)[-1].replace(".nii.gz", "")
        moving_image_name = os.path.split(moving_path)[-1].replace(".nii.gz", "")
        out_dir = os.path.join(REGISTRATION_DIR, f"fixed_{fixed_image_name}", moving_image_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not override and os.path.exists(os.path.join(out_dir, "result.1.nii.gz")):
        print(f"Registration already exists for fixed image {fixed_image_name} and moving image {moving_image_name}. Skipping registration.")
        return

    if fixed_path == moving_path:
        shutil.copy(fixed_path, os.path.join(out_dir, os.path.basename(fixed_path)))
        print(f"Copied fixed image to {out_dir}")
        return

    command = [
        elastix_exe,
        "-f", fixed_path,
        "-m", moving_path,
        "-p", param_affine,
        "-p", param_bspline,
        "-out", out_dir,
    ]

    process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
    process.wait()

    if process.returncode != 0:
        print(f"Error running elastix: {process.returncode}")
        return

    if atlas:
        probability_map_path = moving_path.replace("intensity", "prob")
        probability_map = nib.load(probability_map_path).get_fdata()
        output_3d_files = []
        aff = nib.load(fr"{out_dir}\result.1.nii.gz").affine
        print(f"Affine: {aff}")
        for i in range(probability_map.shape[3]):
            print(f"Processing class {i} of the probability map...")
            class_3d = probability_map[:, :, :, i]
            class_3d_path = os.path.join(out_dir, f"class_{i}.nii.gz")
            nib.save(nib.Nifti1Image(class_3d, aff), class_3d_path)

            transform_param_file = os.path.join(out_dir, "TransformParameters.1.txt")
            # modify_transform_parameters(transform_param_file)

            transform_command = [
                transformix_exe,
                "-in", class_3d_path,
                "-out", out_dir,
                "-tp", transform_param_file,
            ]

            transform_process = subprocess.Popen(transform_command,)# creationflags=subprocess.CREATE_NEW_CONSOLE)
            transform_process.wait()

            if transform_process.returncode != 0:
                print(f"Error transforming class {i}: {transform_process.returncode}")
            else:
                print(f"Class {i} transformed successfully.")
                output_3d_files.append(class_3d_path)

        print("Combining transformed classes into a new 4D probability map...")

        transformed_classes = [nib.load(f).get_fdata() for f in output_3d_files]
        new_probability_map = np.stack(transformed_classes, axis=-1)
        print(transformed_classes[0].shape)
        print(new_probability_map.shape)
        new_probability_map_path = os.path.join(out_dir, "transformed_probability_map.nii.gz")
        nib.save(nib.Nifti1Image(new_probability_map, aff), new_probability_map_path)

        keep_files = {"result.0.nii.gz", "result.1.nii.gz", "result.nii.gz",
                      "TransformParameters.0.txt", "TransformParameters.1.txt",
                      "transformed_probability_map.nii.gz"}
        # for filename in os.listdir(out_dir):
        #     if filename not in keep_files:
        #         file_path = os.path.join(out_dir, filename)
        #         if os.path.isfile(file_path):
        #             print(f"Deleting file: {file_path}")
        #             os.remove(file_path)
        # print("Intermediate files cleaned up.")
    else:
        label_image = moving_path.replace(".nii.gz", "_seg.nii.gz")
        transform_param_file = os.path.join(out_dir, "TransformParameters.1.txt")
        modify_transform_parameters(transform_param_file)

        transform_command = [
            transformix_exe,
            "-in", label_image,
            "-out", out_dir,
            "-tp", transform_param_file,
        ]

        transform_process = subprocess.Popen(transform_command,)# creationflags=subprocess.CREATE_NEW_CONSOLE)
        transform_process.wait()

        if transform_process.returncode != 0:
            print(f"Error running transformix: {transform_process.returncode}")
        else:
            print("Transformix completed successfully.")

        keep_files = {"result.0.nii.gz", "result.1.nii.gz", "result.nii.gz",
                      "TransformParameters.0.txt", "TransformParameters.1.txt"}
        for filename in os.listdir(out_dir):
            if filename not in keep_files:
                file_path = os.path.join(out_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

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

def get_file_paths(base_dir):
    """
    Collect file paths for images and labels from the specified base directory.

    This function traverses the directory structure under the base directory and collects paths
    for image files and their corresponding label files for the Test, Training, and Validation sets.

    :param base_dir: The base directory containing the dataset.
    :return: A dictionary with keys 'Test_Set', 'Training_Set', and 'Validation_Set', each containing
             a dictionary with keys 'img' and 'labels' that map to lists of file paths.
    """
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

def get_dataset_file_paths():
    """
    Get file paths for images and labels in the dataset directory.

    :return: A dictionary with keys 'Test_Set', 'Training_Set', and 'Validation_Set', each containing
             a dictionary with keys 'img' and 'labels' that map to lists of file paths.
    """
    return get_file_paths(DATASET_DIR)

def read_nii(path):
    """
    Read a NIfTI file from the specified path.

    :param path: The path to the NIfTI file.
    :return: The NIfTI image data.
    """
    return nib.load(path).get_fdata()

def get_prob_atlases():
    """
    Get the probability atlases from the registration directory.
    :return: A numpy array of probability atlases.
    """
    paths = collect_paths_with_filename(REGISTRATION_DIR, 'prob_atlas.nii.gz')
    ims = []
    for path in paths:
        ims.append(nib.load(path).get_fdata())
    return np.array(ims), paths

def get_int_atlases():
    """
    Get the intensity atlases from the registration directory.
    :return:
    """
    paths = collect_paths_with_filename(REGISTRATION_DIR, 'intensity_atlas.nii.gz')
    ims = []
    for path in paths:
        ims.append(nib.load(path).get_fdata())
    return np.array(ims), paths
