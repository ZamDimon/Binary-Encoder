import numpy as np
from ..binarer import similarity
from typing import Tuple

def sigma_score(sigma_same: float, sigma_diff: float) -> float:
    """
    sigma_score returns a sigma score based on sigma_same and sigma_diff according to the formula in the paper

    Input:
    sigma_same - average similarity between images of the same person
    sigma_diff - average similarity between images of different people

    Output:
    sigma_score - sigma score
    """

    return np.maximum(((sigma_same-sigma_diff)**(1/3))*sigma_same, 0)

class AccuracyEstimator:
    def __init__(self, X: list, Y: list):
        """
        __init__ initiatilizes an estimator entity with feature vectors X and labels Y
        """
        self.X = X
        self.Y = Y
    
    def estimate(self, converter: callable) -> Tuple[float, int] | Tuple[float, int] | float:
        """
        Estimate is a function that gives estimate based on converter function
        
        Input:
        converter: a function X -> Y

        Output:
        (sigma_same_averaged, same_pairs) - average same similarity and number of pairs
        (sigma_diff_averaged, diff_pairs) - average diff similarity and number of pairs
        sigma_score - sigma score
        """
        
        # Initialize a list of unique classes
        Y_unique = np.unique(self.Y)
        # Initialize a list of images (image from the same batch correspond to the same class)
        X_batches = []

        for c in Y_unique:
            X_batches.append([x for i,x in enumerate(self.X) if self.Y[i] == c])
        
        # Sum of all same sigmas
        sigma_same_total = 0.0
        # Number of entries in the sum (as we would need further calculate average value)
        sigma_same_entries = 0

        for batch in X_batches:
            for j in range(0, len(batch)-1):
                converted_1 = converter(batch[j])
                converted_2 = converter(batch[j+1])
                sigma_same_total += similarity(converted_1, converted_2)
                sigma_same_entries += 1

        # Sum of all diff sigmas
        sigma_diff_total = 0.0
        # Number of entries in the sum
        sigma_diff_entries = 0


        for i in range(len(X_batches)-1):
            current_number_of_pairs = min(len(X_batches[i]), len(X_batches[i+1]))
            for j in range(current_number_of_pairs-1):
                converted_1 = converter(X_batches[i][j])
                converted_2 = converter(X_batches[i+1][j])
                sigma_diff_total += similarity(converted_1, converted_2)
                sigma_diff_entries += 1
        
        # Calculating averaged values
        sigma_same_averaged = sigma_same_total / sigma_same_entries
        print(sigma_same_averaged)
        sigma_diff_averaged = sigma_diff_total / sigma_diff_entries
        print(sigma_diff_averaged)
        score = sigma_score(sigma_same_averaged, sigma_diff_averaged) 
        print(score)

        return (sigma_same_averaged, sigma_same_entries), (sigma_diff_averaged, sigma_diff_entries), score