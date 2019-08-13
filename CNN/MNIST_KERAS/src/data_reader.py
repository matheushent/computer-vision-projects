from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np

class DataReader():
    def __init__(self, data):
        self.data = data

        self.validation_data = (self.data.validation.images, self.data.validation.labels)
        
        self.x = data.train.images
        self.y = data.train.labels