"""
Model class
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.layers import Softmax
from keras.utils import plot_model

from data_reader import LoadData
from hparams import Hparams

import matplotlib.pyplot as plt

class Model():
    def __init__(self):
        # Now we have access of the data and the hyperparameters
        self.data = LoadData()
        self.hparams = Hparams()

    def create_layers(self, model, num_dense_layers,
                      num_dense_nodes, activation):
    
        for i in range(num_dense_layers):
            # Name of the layer. This is not really necessary
            # because Keras should give them unique names.
            name = 'layer_dense_{0}'.format(i+1)

            # Add the dense / fully-connected layer to the model.
            model.add(Dense(num_dense_nodes,
                            name=name))
            model.add(Activation(activation))

    def create_model(self):
        """
        Hyper-parameters:
        learning_rate:     Learning-rate for the optimizer.
        decay:             Decay for the optimizer.
        num_dense_layers:  Number of dense layers.
        num_dense_nodes:   Number of nodes in each dense layer.
        activation:        Activation function for all layers.
        padding:           Padding of the convolutional layers.
        """

        # Start the construction
        model = Sequential()

        model.add(Conv2D(32, self.hparams.training_hparams.kernel_size,
                        padding=self.hparams.training_hparams.padding,
                        input_shape=self.data.x_train.shape[1:]))
        model.add(Activation(self.hparams.training_hparams.activation))
        model.add(Conv2D(32, self.hparams.training_hparams.kernel_size))
        model.add(Activation(self.hparams.training_hparams.activation))
        model.add(MaxPooling2D(pool_size=self.hparams.training_hparams.pool_size))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, self.hparams.training_hparams.kernel_size,
                        padding=self.hparams.training_hparams.padding))
        model.add(Activation(self.hparams.training_hparams.activation))
        model.add(Conv2D(64, self.hparams.training_hparams.kernel_size))
        model.add(Activation(self.hparams.training_hparams.activation))
        model.add(MaxPooling2D(pool_size=self.hparams.training_hparams.pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())

        # Now we create the dense layers
        self.create_layers(model,
                        self.hparams.training_hparams.layers,
                        self.hparams.training_hparams.nodes,
                        self.hparams.training_hparams.activation)
        
        model.add(Dropout(0.5))

        model.add(Dense(self.hparams.image_params.num_classes))
        # Now we will have the probability of all classes
        model.add(Softmax())

        # Initiate RMSprop optimizer and configure some parameters
        optimizer = RMSprop(lr=self.hparams.training_hparams.lr,
                            decay=self.hparams.training_hparams.decay)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        #plot_model(model, to_file='C:/Users/mathe/Desktop/Useful/soloProjects/CNN-GAN/CNN/CIFAR-10/assets/model.png')

        return model