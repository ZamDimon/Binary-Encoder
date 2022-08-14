import face_recognition
import numpy as np
import tensorflow as tf
from PIL import Image


# Function extracted from keras facenet github
def prewhiten(x):
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


def extract_vectors_from_batches(image_batches, model, segment_photo=False):
    """
    Feature vectors extraction from image batches

    Input:
    image_batches -- a batch of images
    model -- Keras Facenet model

    Output:
    vector_batches -- an array of feature vectors for each batch
    """

    vector_batches = []
    for batch_id, batch in enumerate(image_batches):
        print('Loading batch #' + str(batch_id) + '...')
        for image_id, image in enumerate(batch):
            resized_image = tf.image.resize(image, [160, 160])
            resized_image_np = resized_image.eval(session=tf.compat.v1.Session())
            batch[image_id] = resized_image_np

        batch_np = np.array(batch)
        print('Successfully processed batch #' + str(batch_id))

        batch_np = prewhiten(batch_np)
        vector_batch = model.predict_on_batch(batch_np)
        vector_batches.append(vector_batch)
    
    print('Loading complete!')
    return vector_batches


def extract_vectors_from_batches_facenet(image_batches):
    """
    Feature vectors extraction from image batches
    
    Input:
    image_batches -- a batch of images
    model -- Keras Facenet model
    
    Output:
    vector_batches -- an array of feature vectors for each batch
    """
    
    vector_batches = []
    
    for batch in image_batches:
        vector_batch = []
        for person_image in batch:
            encodings = face_recognition.face_encodings(person_image.astype('uint8'))
            if len(encodings) > 0:
                vector_batch.append(encodings[0])
        vector_batches.append(vector_batch)
    
    return vector_batches


def feature_vectors_distance(vector1, vector2):
    """
    Returns distance between two given vectors

    Input:
    vector1, vector2 -- two vectors

    Output:
    distance between vectors
    """
    return np.sum(np.square(vector1 - vector2))


def get_image_features(image, model):
    """
    Function that returns array of features for a given image
    """
    image_resized = tf.image.resize_images(np.expand_dims(image, axis=0), [160, 160])
    image_np = image_resized.eval(session=tf.compat.v1.Session())
    image_np = prewhiten(image_np)
    vector_batch = model.predict_on_batch(image_np)
    return vector_batch[0]


def get_image_features_face_recognition(image):
    """
    Function that returns array of features for a given image
    """
    return face_recognition.face_encodings(image.astype('uint8'))[0]


def get_distance_between_images(image1, image2, model):
    """
    Function that retrieves the distance between two given images (image1 and image2)
    """
    feature_vector_1 = get_image_features(image1, model)
    feature_vector_2 = get_image_features(image2, model)
    return feature_vectors_distance(feature_vector_1, feature_vector_2)

def get_distance_between_image_face_recognition(image1, image2):
    """
    Function that retrieves the distance between two given images (image1 and image2)
    """
    feature_vector_1 = get_image_features_face_recognition(image1)
    feature_vector_2 = get_image_features_face_recognition(image2)
    return feature_vectors_distance(feature_vector_1, feature_vector_2)
    

