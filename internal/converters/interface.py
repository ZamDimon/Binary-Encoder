from typing import List

class IConverter:
    """
    IConverter is the interface that all image-to-vector classes must implement
    """
    
    def convert_to_feature_vector(self, image) -> List[float]:
        """
        convert_to_feature_vector, given image return a feature vector
        """
        pass   