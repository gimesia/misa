import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

flatten = lambda image, binary_mask: image[binary_mask == 1].reshape(-1, 1)


# E-step: Compute memberships with regularization with spatial probabilites
def expectation_step(image_flat, means, variances, mixing_coeffs, num_classes, prob_atlas):
    memberships = np.zeros((image_flat.shape[0], num_classes))

    # Calculate membership for each class
    for c in range(num_classes):
        prob = multivariate_normal(mean=means[c], cov=variances[c], allow_singular=True).pdf(image_flat)
        memberships[:, c] = mixing_coeffs[c] * prob

    # Normalize
    memberships /= memberships.sum(axis=1, keepdims=True)

    memberships = memberships * prob_atlas
    return memberships

def maximization_step(image_flat, memberships, num_classes):
    """
    Perform the maximization step of the EM algorithm.
    
    :param image_flat: Flattened input image (1D numpy array).
    :param memberships: Membership probabilities (2D numpy array).
    :param num_classes: Number of classes.
    :return: Updated means, variances, and mixing coefficients.
    """
    # Reshape memberships to (num_classes, num_voxels)
    memberships = memberships.reshape(num_classes, -1)

    weights = np.sum(memberships, axis=1)
    means = np.dot(memberships, image_flat) / weights[:, np.newaxis]
    variances = np.dot(memberships, (image_flat - means[:, np.newaxis]) ** 2) / weights[:, np.newaxis]
    mixing_coeffs = weights / image_flat.shape[0]

    return means.flatten(), variances.flatten(), mixing_coeffs.flatten()

def initialize_parameters_label_prop(image, atlas, num_classes):
    print("Label propagation initialization")

    # Flatten the image and atlas based on the binary mask
    image_flat = image.reshape(-1)
    labels = np.argmax(atlas, axis=3)
    labels_flat = labels.reshape(-1)

    means = []
    variances = []
    mixing_coeffs = []

    # Loop through each unique label in the atlas (excluding background if label 0 is background)
    for label in range(1, num_classes + 1):  # assuming labels start from 1 for tissues
        class_pixels = image_flat[labels_flat == label]  # Pixels in this class

        if class_pixels.size == 0:
            # If there are no pixels for this class in the image, assign default values
            means.append(0)
            variances.append(1)
            mixing_coeffs.append(0)
            continue

        # Calculate mean and variance for this class
        mean = np.mean(class_pixels)
        variance = np.var(class_pixels)

        # Mixing coefficient as the proportion of pixels in this class
        mixing_coeff = class_pixels.size / image_flat.size

        means.append(mean)
        variances.append(variance)
        mixing_coeffs.append(mixing_coeff)

    print("Means:", means)
    print("Variances:", variances)
    print("Mixing Coefficients:", mixing_coeffs)

    return np.array(means), np.array(variances), np.array(mixing_coeffs)

# Initialize using K-means with regularization
def initialize_parameters_kmeans(image_flat, num_classes):
    # Use K-means to initialize
    kmeans = KMeans(n_clusters=num_classes, random_state=42).fit(image_flat)
    means = kmeans.cluster_centers_.flatten()

    print("K-means initialized means:")
    print(means)

    # Compute initial variances and apply regularization
    variances = np.array([np.var(image_flat[kmeans.labels_ == i]) for i in range(num_classes)])
    variances = np.maximum(variances, 1e-6)  # Regularize variances (ensure they are not too small)

    mixing_coeffs = np.array([np.sum(kmeans.labels_ == i) for i in range(num_classes)]) / len(image_flat)

    return means, variances, mixing_coeffs


# GMM segmentation for 3D image data with regularization
def em_atlas_segmentation(image, atlas, num_classes=4, max_iters=100, tol=1e-3):
    # Flatten the 3D image
    mask = image == 0
    image_flat = image.reshape(-1, 1)
    atlases = np.array((
        atlas[:, :, :, 0].reshape(-1,1),
        atlas[:, :, :, 1].reshape(-1,1),
        atlas[:, :, :, 2].reshape(-1,1),
        atlas[:, :, :, 3].reshape(-1,1)
    )).T

    # Initialize using label propagation
    # means, variances, mixing_coeffs = initialize_parameters_label_prop(image_flat, atlas, num_classes)
    means, variances, mixing_coeffs = initialize_parameters_kmeans(image_flat, num_classes)

    for iteration in range(max_iters):
        if iteration % 25 == 0:
            print(f"E-step; iteration: {iteration}/{max_iters}")
        memberships = expectation_step(image_flat, means, variances, mixing_coeffs, num_classes, atlases)

        new_means, new_variances, new_mixing_coeffs = maximization_step(image_flat, memberships, num_classes)

        # Check convergence
        if np.allclose(means, new_means, atol=tol) and np.allclose(variances, new_variances, atol=tol):
            print(f"Converged at iteration {iteration}")
            break

        means, variances, mixing_coeffs = new_means, new_variances, new_mixing_coeffs
        if iteration == max_iters - 1:
            print(f"Reached maximum iterations {max_iters}")

    # Final segmentation: Assign each voxel to the class with the highest membership
    segmentation = np.argmax(memberships, axis=-1)

    # Reshape segmentation back to 3D
    segmentation_3d = np.zeros_like(image)

    return segmentation, (means, variances, mixing_coeffs)


def atlas_seg(atlas):
    return np.argmax(atlas)



