from warnings import filterwarnings
filterwarnings('ignore')

import matplotlib.pyplot as plt

class Plot():
    def __init__(self, history):
        self.history = history
        self.acc_path = 'C:/Users/mathe/Desktop/CNN-GAN/CNN/MNIST_KERAS/assets/acc_score.png'
        self.loss_path = 'C:/Users/mathe/Desktop/CNN-GAN/CNN/MNIST_KERAS/assets/loss_score.png'
        self.format = 'png'
        self.legend = ['Train', 'Test']

    def clear_fig(self):
        plt.clf()

    def plot_acc(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(self.legend, loc='upper left')
        plt.savefig(self.acc_path, format=self.format)

    def plot_loss(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(self.legend, loc='upper left')
        plt.savefig(self.loss_path, format=self.format)