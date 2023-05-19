from .main import IDataset

class CelebaLoader(IDataset):
    def __init__(self, images_root, txt_path):
        self.images_root = images_root
        self.txt_path = txt_path
        pass

    def load(self):
        """
        load returns two lists: set of paths and corresponding labels
        Output:
        image_paths - set of image paths
        labels - set of person IDs (that is, labels)
        """

        with open(self.txt_path) as txt_file:
            txt_file_lines = txt_file.read().splitlines()
        
        image_paths = [self.images_root + '/' + labels_file_line.split(" ")[0] for labels_file_line in txt_file_lines]
        labels = [labels_file_line.split(" ")[1] for labels_file_line in txt_file_lines]

        return image_paths, labels