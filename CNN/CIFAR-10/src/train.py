from warnings import filterwarnings
filterwarnings('ignore')

from model import Model
from callbacks import Callbacks
from graphics import PlotMetrics

from keras.utils import plot_model

class FitModel():
    def __init__(self):
        self.model = Model()
        self.Model = self.model.create_model()
        self.callback = Callbacks()
    
    def fit(self):
        self.history = self.Model.fit(x=self.model.data.x_train,
                                      y=self.model.data.y_train,
                                      epochs=self.model.data.params.training_hparams.epochs,
                                      batch_size=self.model.data.params.training_hparams.batch_size,
                                      validation_data=self.model.data.validation_data,
                                      callbacks=[self.callback.tensorboard.callback_log])
        return self.history

trained = FitModel()
history = trained.fit()

metrics_plotter = PlotMetrics(history)

metrics_plotter.plot_acc()
metrics_plotter.clear_fig()
metrics_plotter.plot_loss()

trained.Model.save('C:/Users/mathe/Desktop/CNN-GAN/CNN/CIFAR-10/logs/model.h5')
plot_model(trained.Model, to_file='C:/Users/mathe/Desktop/CNN-GAN/CNN/CIFAR-10/assets/model.png')