import numpy as np
from helpers import vector_extractor


def calculate_pairs_binary_accuracy(pairs, feature_vector_batches, threshold=100):
    """
    Calculates cumulative accuracy by calculating number
    of successfull recognitions to the total number of guesses

    Input:
    pairs -- two-dimensional array each element of which corresponds to [n1, i1, n2, i2]
    feature_vector_batches -- array of feature vectors

    Output:
    accuracy -- cumulative accuracy obtained by method described above
    """

    accuracies = np.empty((1 ,0))

    for pair in pairs:
        n1, i1, n2, i2 = pair
        distance = vector_extractor.feature_vectors_distance(feature_vector_batches[n1][i1], feature_vector_batches[n2][i2])
        if distance < threshold:
            accuracies = np.append(accuracies, float(n1 == n2))
        else:
            accuracies = np.append(accuracies, float(n1 != n2))

    return np.mean(accuracies)


def threshold_accuracies(pairs, feature_vectors, lower_edge=40.0, upper_edge=300.0, step=0.1):
    """
    Function that returns a set of pairs [threshold, binary accuracy for this threshold]

    Inputs:
    pairs, feature_vectors -- a set of pairs of form [n1, i1, n2, i2] and a set of feature vectors
    lower_edge, upper_edge -- a value to begin with and iterate until, respectively
    step -- size of step

    Outputs:
    threshold -- a vector with all thresholds the function iterates through
    accuracies -- a vector with all corresponding cumulative accuracies
    """

    thresholds, accuracies = [], []
    for threshold in np.arange(lower_edge, upper_edge, step):
        thresholds.append(threshold)
        accuracies.append(calculate_pairs_binary_accuracy(pairs, feature_vectors, threshold=threshold))

    return thresholds, accuracies


def get_best_threshold(pairs, feature_vectors, lower_edge=40.0, upper_edge=300.0, step=0.01):
    """
    Function that returns the threshold with the best binary accuracy

    Inputs and outputs:
    Same as for threshold_accuracies function

    """
    thresholds, accuracies = threshold_accuracies(pairs, feature_vectors,
                                                  lower_edge=lower_edge, upper_edge=upper_edge, step=step)
    maximum_accuracy_number = np.argmax(accuracies)
    return thresholds[maximum_accuracy_number]


def calculate_accuracy(same, distance, threshold=100.0, eta_1=3.0, eta_2=3.0):
    """
    Calculates accuracy of distance evaluation based on given distance and whether
    two images are from one batch

    Inputs:
    same - boolean that equals to True if two images are of a same person and False if vice versa
    distance - distance between two images
    For meaning of other parameters see block above

    Output:
    accuracy - calculated value based on the block above
    """
    return max(1 - pow(distance / threshold, eta_1), 0) if same == True else max(1 - pow(threshold / distance, eta_2), 0)


def calculate_pair_accuracy(feature_vectors, n1, i1, n2, i2, threshold=100, eta_1=3.0, eta_2=3.0):
    """
    Calculates accuracy of distance evaluation between two given pages

    Inputs:
    feature_vectors - set of vector batches of shape (n_b, n_i,)
    n1, n2 - number of batch of the first and second image, respectively
    i1, i2 - number of images in the first (n1) and second (n2) batch, respectively

    Output:
    accuracy - calculated value based on the block above
    """

    distance = vector_extractor.feature_vectors_distance(feature_vectors[n1][i1], feature_vectors[n2][i2])
    return calculate_accuracy(n1 == n2, distance, threshold=threshold, eta_1=eta_1, eta_2=eta_2)


def calculate_pairs_accuracy(pairs, feature_vector_batches, threshold=100):
    """
    Calculates cumulative accuracy after comparing multiple pairs of vectors

    Input:
    pairs -- two-dimensional array each element of which corresponds to [n1, i1, n2, i2]
    feature_vector_batches -- array of feature vectors

    Output:
    accuracy -- cumulative accuracy
    """
    accuracies = np.empty((1, 0))

    for pair in pairs:
        n1, i1, n2, i2 = pair
        accuracies = np.append(accuracies,
                               calculate_pair_accuracy(feature_vector_batches, n1, i1, n2, i2, threshold=threshold))

    return np.sqrt(np.mean(accuracies ** 2))
