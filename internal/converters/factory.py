from typing import List
from enum import Enum

from .interface import IConverter
from .keras_facenet import KerasFacenetConverter
from .face_recognition import FaceRecognitonConverter

class ConverterType(Enum):
    FACE_RECOGNITION = 1
    KERAS_FACENET = 2

class ConverterFactory():
    def __init__(self, model=None):
        self.converters = {
            ConverterType.KERAS_FACENET: KerasFacenetConverter(model),
            ConverterType.FACE_RECOGNITION: FaceRecognitonConverter()
        }

    def get_converter(self, converter: ConverterType) -> IConverter:
        return self.converters[converter]