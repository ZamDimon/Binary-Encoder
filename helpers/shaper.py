import numpy as np
INFTY = 10000000


def form_batch_by_id(images, labels, image_id):
    """
    Get array of images of a person with specified id from the dataset

    Input:
    images - array of images of shape (m, n_H, n_W, n_C)
    labels - array of labels (integers)
    image_id - id of a desired set of images

    Output:
    image_batch - array of images of a person with id=person_id of shape (k, n_H, n_W, n_C)
    """

    _, image_height, image_width, image_channels = images.shape
    image_batch = np.empty((0, image_height, image_width, image_channels))

    assert len(labels) == images.shape[0], 'Labels length should match the number of images'

    for i in range(len(labels)):
        if labels[i] == image_id:
            image_to_concatenate = np.expand_dims(images[i], axis=0)
            image_batch = np.concatenate((image_batch, image_to_concatenate), axis=0)

    return image_batch


def form_batches_list(images, labels):
    """
    Form set of batches in which the images correspond to one single ID

    Input:
    images - set of images of shape (m, n_H, n_W, n_C)
    labels - set of labels of size m

    Output:
    image_batches - batches of images with the same id (which equals to the index in an array)
    """

    image_batches = []
    for person_id in np.unique(labels):
        image_batches.append(form_batch_by_id(images, labels, person_id))

    return image_batches


def form_batches_numpy(images, labels, dataset='sklearn_lfw_people'):
    """
    Same as form_batches_list, but creates a batch of shape (m, k, n_H, n_W, n_C)
    where m is a number of batches, k is a minimum size among all batches

    Input:
    images, labels -- same as for form_batches_list
    dataset -- either 'local_lfw_people' or 'sklearn_lfw_people'

    Output: a set of shape (m, k, n_H, n_W, n_C)
    """

    if dataset == 'local_lfw_people':
        # As the size of images batches is already normalize, we just simply convert it
        return np.array(images)
    elif dataset == 'sklearn_lfw_people':
        # Form list of batches
        batches_list = form_batches_list(images, labels)

        # Get minimum size of a batch
        min_batch_size = INFTY
        for batch in batches_list:
            min_batch_size = min(min_batch_size, batch.shape[0])

        _, image_height, image_width, image_channels = np.shape(images)
        batches_list_numpy = np.empty((0, min_batch_size, image_height, image_width, image_channels))

        print('Concatenating...')
        for batch_id, batch in enumerate(batches_list):
            print('Trying to concatenate batch #' + str(batch_id) + '...')
            batch_to_concatenate = np.expand_dims(batch[:min_batch_size], axis=0)
            batches_list_numpy = np.concatenate((batches_list_numpy, batch_to_concatenate), axis=0)
            print('Suceess!')

        print('Loading completed')
        return batches_list_numpy

    return None


def normalize_batches_size(image_batches):
    """
    Makes the batch of the single size
    """

    normalized_batches = []
    # Get minimum size of a batch
    min_batch_size = INFTY
    for batch in image_batches:
        min_batch_size = min(min_batch_size, np.shape(batch)[0])

    for batch in image_batches:
        normalized_batches.append(batch[:min_batch_size])

    return normalized_batches


def get_minimal_batch_size(feature_vectors):
    minimal_batch_size = 10000000
    for batch in feature_vectors:
        minimal_batch_size = min(minimal_batch_size, len(batch))

    return minimal_batch_size


def form_pairs_list(feature_vectors_shape, fixed_size=None):
    """
    Forms list of pairs of shape (n_p, 4) where n_p is a number of pairs and
    each element corresponds to the numbers (n_1, i_1, n_2, i_2)

    Input:
    feature_vectors_shape - shape of feature_vectors array

    Output: See description of the function
    """

    pairs = []
    pairs_number_in_row = int(np.floor(feature_vectors_shape[1] / 2))

    for batch in range(feature_vectors_shape[0] - 1):
        for i in range(pairs_number_in_row):
            pairs.append([batch, 2 * i, batch, 2 * i + 1])
        for i in range(pairs_number_in_row):
            pairs.append([batch, i, batch + 1, i])

    return pairs


def split_pairs(pairs):
    """
    Splits the pairs array into two arrays

    Output:
    same_pairs, different_pairs -- pairs of the same person and of two different ones, respectively
    """

    same_pairs, different_pairs = [], []
    for pair in pairs:
        n1, _, n2, _ = pair
        if n1 == n2:
            same_pairs.append(pair)
        else:
            different_pairs.append(pair)

    return same_pairs, different_pairs
