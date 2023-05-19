# Global imports
import csv
import numpy as np
import cv2
from typing import List

# Local imports
from .main import IDataset

class CSV(IDataset):
    """
    Dataseter is responsible for loading and saving datasets
    """

    def __init__(self, input_path=None, output_path=None):
        self.input_path = input_path
        self.output_path = output_path

    def read(self):
        """
        read returns a set of feature vectors and labels from the dataset
        
        Output:
        X - set of feature vectors
        Y - set of labels
        links - links to images
        """

        X, Y, links = [], [], []
        with open(self.input_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                # Omitting the table header
                if i == 0:
                    continue

                # Third column is a column with person id
                label = int(float(row[2]))
                vector = np.array(row[3:]).astype(np.float)

                X.append(vector)
                Y.append(label)
                links.append(row[1])
        return X, Y, links

    def form_row_csv(self, entry_id: int, path: str, person_id: int, feature_vector: list, precision: int = None):
        """
        Returns an array that corresponds to a single row in the csv file

        Input:
        entry_id - id of entry
        path - path to the image
        person_id - id of a person
        feature_vector - feature vector of this person
        precision - number of digits after a comma to be inserted (None if leave precision as it is)

        Output:
        Row of a csv table 
        """
        
        if precision != None:
            feature_vector = (feature_vector*(10**precision)).astype(int)/(10**precision)
        
        return [entry_id, path, person_id, *feature_vector]

    def load_csv(self, image_paths: List[str], labels: List[str], converter: callable, verbose: bool=True):
        """
        Loads converted vectors into the CSV file
        
        Input:
        image_paths - list of paths to the desired images
        labels - list of labels
        converter - function that converts image to a vector 
        verbose - whether to log how the loading is being processed
        """

        with open(self.output_path, 'a') as csv_file:
            number_of_images = len(image_paths)
            number_of_processed_images = 0
            logging_frequency = 1000
            
            writer = csv.writer(csv_file)
            
            for i in range(number_of_images):
                path = image_paths[i]
                image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                
                encoding = converter(image)
                if encoding is None:
                    continue
                
                if number_of_processed_images % logging_frequency == 0 and verbose:
                    print('Successfully processed {} images'.format(number_of_processed_images))
                    
                number_of_processed_images += 1
                row = self.form_row_csv(i, path, labels[i], encoding, precision=None)
                writer.writerow(row)