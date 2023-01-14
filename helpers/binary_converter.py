import numpy as np

def form_string_batches(feature_vectors):
    """
    Function that converts the feature vector batches to the string batches

    Input:
    feature_vectors -- array of feature vectors batches

    Output:
    string_batches -- array of string batches of the same shape as feature_vectors array
    """

    string_batches = []

    for batch in feature_vectors:
        string_batch = []
        for feature_vector in batch:
            string_batch.append(form_binary_string(feature_vector))

        string_batches.append(string_batch)

    return string_batches

def form_string_batches_with_offset(feature_vectors, offset=0.0):
    """
    Function that converts the feature vector batches to the string batches

    Input:
    feature_vectors -- array of feature vectors batches

    Output:
    string_batches -- array of string batches of the same shape as feature_vectors array
    """

    string_batches = []

    for batch in feature_vectors:
        string_batch = []
        for feature_vector in batch:
            string_batch.append(form_binary_string_with_offset(feature_vector, offset))

        string_batches.append(string_batch)

    return string_batches

def form_binary_string(feature_vector):
    """
    Function that forms the binary string according to the rule described above
    """

    binary_string = ''
    for vector_element in feature_vector:
        binary_string += ('0' if vector_element <= 0 else '1')

    return binary_string

def form_binary_string_with_offset(feature_vector, offset=0.0):
    """
    Function that forms the binary string according to the rule described above
    """

    binary_string = ''
    for vector_element in feature_vector:
        binary_string += ('0' if vector_element <= offset else '1')

    return binary_string

def binary_string_difference(string_1, string_2):
    """
    Function that returns the binary string difference according to the rule described above
    """

    assert len(string_1) == len(string_2), 'strings length must match'

    total_difference = 0
    for i in range(len(string_1)):
        total_difference += abs(int(string_1[i]) - int(string_2[i]))

    return total_difference / len(string_1)


def binary_string_similarity(string_1, string_2):
    """
    Function that returns the binary string difference according to the rule described above
    """

    assert len(string_1) == len(string_2), 'strings length must match'

    return 1.0 - binary_string_difference(string_1, string_2)


def get_average_similarity(string_batches, pairs):
    """
    Returns average similarity across given pairs
    """

    similarities = []

    for pair in pairs:
        n1, i1, n2, i2 = pair
        similarities.append(binary_string_similarity(string_batches[n1][i1], string_batches[n2][i2]))

    return np.mean(similarities)
