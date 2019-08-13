from warnings import filterwarnings
filterwarnings('ignore')

from tensorflow.python.keras.layers import Dense

from metadata_generator import Generator

def create_layers(model, num_dense_layers,
                num_dense_nodes, activation):
    
    for i in range(num_dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)

        # Add the dense / fully-connected layer to the model.
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name))

class TensorBoardConfig():
    def __init__(self, learning_rate,
                activation,
                num_dense_nodes,
                num_dense_layers,
                FLAGS, data):

        self.flags = FLAGS
        self.data = data
        
        self.learning_rate = learning_rate
        self.num_dense_layers = num_dense_layers
        self.num_dense_nodes = num_dense_nodes
        self.activation = activation
        
        #Generator(self.flags,
        #          self.data)
        
        self.metadata = open('C:/Users/mathe/Desktop/CNN-GAN/CNN/MNIST_KERAS/logs/projector/metadata.tsv',
                            errors='ignore', encoding='utf-8').read()

        # The dir-name for the Tensorboard log-dir
        s = "C:/Users/mathe/Desktop/CNN-GAN/CNN/MNIST_KERAS/logs/projector/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"

        # Insert all the hyper-parameters in the dir-name
        self.log_dir = s.format(self.learning_rate,
                                self.num_dense_layers,
                                self.num_dense_nodes,
                                self.activation)