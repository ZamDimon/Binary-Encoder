from .interface import IConverter
import face_recognition
from typing import List

class FaceRecognitonConverter(IConverter):
    """
    FaceRecognitionConverter is a class implementing IConverter 
    that using face recognition model converts image to a feature vector
    """
    def __init__(self):
        pass

    def convert_to_feature_vector(self, image) -> List[float]:
        """
        convert_to_feature_vector returns a feature vector of length 128 from an image
        If either no or more than one image was found on the image, returns None
        """
        encodings = face_recognition.face_encodings(image.astype('uint8'))
        if len(encodings) != 1:
            return None

        return encodings[0] 