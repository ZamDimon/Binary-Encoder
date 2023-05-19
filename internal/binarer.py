import numpy as np
import tensorflow as tf

def binary_distance(x, y):
    """
    binary_distance returns a binary distance between two vectors
    
    Input:
    x, y - two vectors of the same size
    
    Output:
    Binary distance between these two vectors
    """
    return tf.reduce_mean(tf.abs(x-y),axis=0)

def to_binary_string(f, t):
    """
    to_binary_string converts given vector to a binary string using rule I(f>t)
    
    Input:
    f - vector
    t - threshold vector
    
    Output:
    binary string of the same length as f and t
    """
    return tf.where(tf.less(f, t), 0.0, 1.0)

def similarity(x, y):
    """
    similarity returns similarity between two given vectors
    
    Input:
    x, y - two vectors of the same size
    
    Output:
    Similarity between these two vectors
    """
    return 1 - binary_distance(x, y)