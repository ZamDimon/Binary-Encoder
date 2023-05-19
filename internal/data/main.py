class IDataset:
    """
    IDataset is the interface that any dataset must implement
    """
    def load(self):
        """
        load returns two lists: set of features and set of labels

        Input: Nothing (class object implementing this interface must have a constructor with all needed parameters)
        Output:
        X - set of features
        Y - set of labels
        """
        pass