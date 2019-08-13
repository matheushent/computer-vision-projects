"""
Hyperparameters class
"""

class Hparams(object):

    def __init__(self):
        self.training_hparams = TrainingHparams()
        self.image_params = ImageParams()

class TrainingHparams(object):
    
    def __init__(self):
        self.learning_rate = 0.004

        self.layers = 1

        self.nodes = 460

        self.activation = 'relu'

        self.batch_size = 128

        self.epochs = 3

        self.kernel_size = 5

        self.conv_strides = 1

        self.pool_strides = 2

        self.pool_size = 2

class ImageParams(object):

    def __init__(self):
        self.img_size = 28

        self.img_size_flat = self.img_size * self.img_size

        self.img_shape = (self.img_size, self.img_size)

        self.img_shape_full = (self.img_size, self.img_size, 1)

        self.num_channels = 1

        self.num_classes = 10