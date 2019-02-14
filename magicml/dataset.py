import pandas as pd
import numpy as mp

class ImportDataset:
    """ reading dataset """
    def __init__(self):
        pass
    
    @staticmethod
    def from_csv(path):
        return pd.read_csv(path)
    
    @staticmethod
    def random(shape):
        return np.random.sample(shape)
