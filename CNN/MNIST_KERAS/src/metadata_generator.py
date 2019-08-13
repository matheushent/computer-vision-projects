"""
This code seems a little bit confuded in the first time, even to me, but it works.
This is the life of a programmer: you don't know why something doesn't works and
why something works. :)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from warnings import filterwarnings
filterwarnings('ignore')

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Generator():
    def __init__(self, FLAGS, data):
        self.data = data
        self.FLAGS = FLAGS
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--max_steps', type=int, default=10000,
                            help='Number of steps to run trainer.')
        self.parser.add_argument('--log_dir', type=str, default='C:/Users/mathe/Desktop/CNN-GAN/CNN/MNIST_KERAS/logs',
                            help='Summaries log directory')
        self.FLAGS, unparsed = self.parser.parse_known_args()
        tf.app.run(main=self.main, argv=[sys.argv[0]] + unparsed)

    def generate_metadata_file(self):

        def save_metadata(file):
            with open(file, 'w') as f:
                for i in range(self.FLAGS.max_steps):
                    c = np.nonzero(self.data.test.labels[::1])[1:][0][i]
                    f.write('{}\n'.format(c))

        save_metadata(self.FLAGS.log_dir + '/projector/metadata.tsv')

    def main(self, _):
        if tf.gfile.Exists(self.FLAGS.log_dir + '/projector'):
            tf.gfile.DeleteRecursively(self.FLAGS.log_dir + '/projector')
            tf.gfile.MkDir(self.FLAGS.log_dir + '/projector')
        tf.gfile.MakeDirs(self.FLAGS.log_dir  + '/projector') # fix the directory to be created
        self.generate_metadata_file()