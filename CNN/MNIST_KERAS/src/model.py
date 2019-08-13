from warnings import filterwarnings
filterwarnings('ignore')
import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import plot_model

from hparams import Hparams
from secondary_functions import create_layers, TensorBoardConfig
from data_reader import DataReader

class CNNModel():
    def __init__(self, FLAGS, data):
        
        self.params = Hparams()
        self.flags = FLAGS

        self.learning_rate = self.params.training_hparams.learning_rate
        self.num_dense_layers = self.params.training_hparams.layers
        self.num_dense_nodes = self.params.training_hparams.nodes
        self.activation = self.params.training_hparams.activation
        
        self.data = DataReader(data)

        self.tensorboard = TensorBoardConfig(self.learning_rate,
                                            self.activation,
                                            self.num_dense_nodes,
                                            self.num_dense_layers,
                                            self.flags,
                                            self.data.data)

    def create_model(self):
        """
        Hyper-parameters:
        learning_rate:     Learning-rate for the optimizer.
        num_dense_layers:  Number of dense layers.
        num_dense_nodes:   Number of nodes in each dense layer.
        activation:        Activation function for all layers.
        """
    
        # Start the construction of the model
        model = Sequential()

        # Add an input layer which is similar to a feed_dict in TensorFlow.
        # Note that the input-shape must be a tuple containing the image-size.
        model.add(InputLayer(input_shape=(self.params.image_params.img_size_flat,)))

        # The input from MNIST is a flattened array with 784 elements,
        # but the convolutional layers expect images with shape (28, 28, 1)
        model.add(Reshape(self.params.image_params.img_shape_full))

        # First convolutional layer.
        model.add(Conv2D(kernel_size=self.params.training_hparams.kernel_size,
                            strides=self.params.training_hparams.conv_strides,
                            filters=16, padding='same',
                            activation=self.params.training_hparams.activation,
                            name='layer_conv1'))
        model.add(MaxPooling2D(pool_size=self.params.training_hparams.pool_size,
                                    strides=self.params.training_hparams.pool_strides))

        # Second convolutional layer.
        model.add(Conv2D(kernel_size=self.params.training_hparams.kernel_size,
                            strides=self.params.training_hparams.conv_strides,
                            filters=36, padding='same',
                            activation=self.params.training_hparams.activation,
                            name='layer_conv2'))
        model.add(MaxPooling2D(pool_size=self.params.training_hparams.pool_size,
                                    strides=self.params.training_hparams.pool_strides))

        # Flatten the 4-rank output of the convolutional layers
        # to 2-rank that can be input to a fully-connected / dense layer.
        model.add(Flatten())

        # Add fully-connected / dense layers.
        create_layers(model,
                    self.num_dense_layers,
                    self.num_dense_nodes,
                    self.activation)

        # Last fully-connected / dense layer with softmax-activation
        # for use in classification.
        model.add(Dense(self.params.image_params.num_classes,
                            activation='softmax'))

        # Use the Adam method for training the network.
        optimizer = Adam(lr=self.learning_rate)

        # In Keras we need to compile the model so it can be trained.
        model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        return model

    def fitness(self):
        print()
        print("Hyper-parameters:")
        """
        Hyper-parameters:
        learning_rate:     Learning-rate for the optimizer.
        num_dense_layers:  Number of dense layers.
        num_dense_nodes:   Number of nodes in each dense layer.
        activation:        Activation function for all layers.
        """

        # Print the hyper-parameters.
        print()
        print('learning rate: {0:.1e}'.format(self.learning_rate))
        print('num_dense_layers:', self.num_dense_layers)
        print('num_dense_nodes:', self.num_dense_nodes)
        print('activation:', self.activation)
        print()

        # Create the neural network with these hyper-parameters.
        self.model = self.create_model()

        # Save an image containing info about the model
        plot_model(self.model, to_file='C:/Users/mathe/Desktop/CNN-GAN/CNN/MNIST_KERAS/assets/model.png')

        # Instantiate TensorBoard class, which
        # will give us access for all what is 
        # happening with our model
        self.callback_log = TensorBoard(log_dir=self.tensorboard.log_dir,
                                        histogram_freq=1,
                                        batch_size=32,
                                        write_graph=True,
                                        write_images=True,
                                        embeddings_metadata=self.tensorboard.metadata)

        # Use Keras to train the model.
        print("\n\n")
        print('Training...')
        print("\n\n")
        self.history = self.model.fit(x=self.data.x,
                                      y=self.data.y,
                                      epochs=self.params.training_hparams.epochs,
                                      batch_size=self.params.training_hparams.batch_size,
                                      validation_data=self.data.validation_data,
                                      callbacks=[self.callback_log])

        return self.history, self.model