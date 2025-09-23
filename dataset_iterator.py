import pandas as pd

class DatasetIterator:
    def __init__(self, annotation_file, class_label):
        self.df = pd.read_csv(annotation_file)
        self.class_label = class_label
        self.instances = self.df[self.df["Class"] == class_label]["Absolute Path"].tolist()

    def __iter__(self):
        self.index = 0 
        return self

    def __next__(self):
        if self.index >= len(self.instances):
            raise StopIteration
        next_path = self.instances[self.index]
        self.index += 1
        return next_path