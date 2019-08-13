from warnings import filterwarnings
filterwarnings('ignore')

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('mnist/', one_hot = True)

from model import CNNModel
from plot import Plot

# It is necessary to call the embedding_config
FLAGS = None

# Instantiate the CNNModel class
CNNModel = CNNModel(FLAGS, data)

history, model = CNNModel.fitness()

# Instantiate the Plot class from plot.py
Plot = Plot(history)

# Save the model
model.save('C:/Users/mathe/Desktop/CNN-GAN/CNN/MNIST_KERAS/logs/model.h5',
            include_optimizer=True,
            overwrite=True)

# Plot training & validation accuracy values
Plot.plot_acc()

# Clear the current figure
Plot.clear_fig()

# Plot training & validation loss values
Plot.plot_loss()