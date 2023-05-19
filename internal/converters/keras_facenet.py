from .interface import IConverter
from typing import List
import numpy as np
import tensorflow as tf

class KerasFacenetConverter(IConverter):
    """
    FaceRecognitionConverter is a class implementing IConverter 
    that using face recognition model converts image to a feature vector
    """
    def __init__(self, model):
        self.model = model

    # Function extracted from keras facenet github
    def prewhiten(self, x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0 / np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    def convert_to_feature_vector(self, image) -> List[float]:
        """
        Function that returns array of features for a given image
        """

        image_resized = tf.image.resize_images(np.expand_dims(image, axis=0), [160, 160])
        image_np = image_resized.eval(session=tf.compat.v1.Session())
        image_np = self.prewhiten(image_np)
        vector_batch = self.model.predict_on_batch(image_np)
        return vector_batch[0]