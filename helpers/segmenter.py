import numpy as np
import face_recognition
from PIL import Image

INFTY = 10000000


def segment_image(image):
    image = image.astype("uint8")
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return 'SEGMENT_ERROR'

    top, right, bottom, left = face_locations[0]
    return np.array(Image.fromarray(image).crop((left, top, right, bottom)))


def segment_batches(image_batches):
    """
    Return the same batch, but with segmented images
    """

    new_batches = []

    for batch_id, batch in enumerate(image_batches):
        new_batch = []
        print('Trying to load batch #' + str(batch_id) + '...')

        for image in batch:
            new_image = segment_image(image)
            if new_image != 'SEGMENT_ERROR':
                new_batch.append(segment_image(image))

        new_batches.append(new_batch)

    return new_batches
