from warnings import filterwarnings
filterwarnings('ignore')

import os

from tensorflow.python.keras.models import load_model

class LoadModel():
    def __init__(self, path):
        self.path = path
        # self.FLAGS = FLAGS
        # self.data = data
        self.exist_file = os.path.isfile(self.path)

    def load_model(self):
        if self.exist_file:
            return load_model(self.path)

test = LoadModel('../assets/acc_score.png')
print(test.exist_file)