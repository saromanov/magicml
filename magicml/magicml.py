import tensorflow as tf

class TextClassification:
    """ Applying of the text classification"""
    def __init__(self, dataset):
        self._dataset = dataset
        self.train_df = load_dataset(os.path.join(os.path.dirname(dataset),"aclImdb", "train"))
        self.test_df = load_dataset(os.path.join(os.path.dirname(dataset),"aclImdb", "test"))