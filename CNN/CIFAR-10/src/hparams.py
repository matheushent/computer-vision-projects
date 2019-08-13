"""
Hyperparameters class
"""
class Hparams(object):

    def __init__(self):
        self.training_hparams = TrainingHparams()
        self.image_params = ImageParams()

class TrainingHparams(object):
    
    def __init__(self):
        self.lr = 0.0001

        self.decay = 1e-6

        self.padding = 'same'

        self.layers = 1

        self.nodes = 512

        self.activation = 'relu'

        self.batch_size = 128

        self.epochs = 1

        self.kernel_size = 3

        self.pool_size = 2

class ImageParams(object):

    def __init__(self):
        self.img_size = 32

        self.num_channels = 1

        self.num_classes = 10