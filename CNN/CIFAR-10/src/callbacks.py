from hparams import Hparams

from keras.callbacks import TensorBoard

class Callbacks():
    def __init__(self):
        self.tensorboard = TensorBoardConfig()

class TensorBoardConfig(object):
    def __init__(self):
        self.hparams = Hparams().training_hparams

        self.metadata = open('C:/Users/mathe/Desktop/CNN-GAN/CNN/CIFAR-10/logs/projector/metadata.tsv',
                            errors='ignore', encoding='utf-8').read()
        
        self.log_dir_name = self.log_dir()

        # Instantiate TensorBoard class, which
        # will give us access for all what is 
        # happening with our model
        self.callback_log = TensorBoard(log_dir=self.log_dir_name,
                                        histogram_freq=1,
                                        batch_size=32,
                                        write_graph=True,
                                        write_images=True,
                                        embeddings_metadata=self.metadata)

    def log_dir(self):

        # The dir-name for the Tensorboard log-dir
        s = "C:/Users/mathe/Desktop/CNN-GAN/CNN/CIFAR-10/logs/projector/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"

        # Insert all the hyper-parameters in the dir-name
        self.log_dir = s.format(self.hparams.lr,
                                self.hparams.layers,
                                self.hparams.nodes,
                                self.hparams.activation)

        return self.log_dir