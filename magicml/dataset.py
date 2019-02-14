import pandas as pd

class ImportDataset:
    """ reading dataset """
    def __init__(self):
        pass
    
    @staticmethod
    def from_csv(path):
        return pd.read_csv(path)
    
    