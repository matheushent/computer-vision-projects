"""
The CIFAR-10 Dataset contains 10 categories of images
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck
"""

import keras
from keras.datasets import cifar10

from hparams import Hparams

class LoadData():
    def __init__(self):
        # Instantiate Hparams class to access the parameters
        self.params = Hparams()

        train, test = cifar10.load_data()

        # Format our training data by normalizing and changing data type
        self.x_train = (train[0].astype('float32'))/255
        x_test = (test[0].astype('float32'))/255

        # Now we hot encode outputs
        self.y_train = keras.utils.to_categorical(train[1],
                                                self.params.image_params.num_classes)
        y_test = keras.utils.to_categorical(test[1],
                                                self.params.image_params.num_classes)

        self.validation_data = (x_test, y_test)