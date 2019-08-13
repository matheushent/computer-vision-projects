from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('mnist/', one_hot=False)

X_train = mnist.train.images
y_train = mnist.train.labels

X_test = mnist.test.images
y_test = mnist.test.labels

y_train = np.asarray(y_train, dtype=np.int32)
y_test = np.asarray(y_test, dtype=np.int32)

X_image_test = X_test[0].reshape(-1, 1)

import matplotlib.pyplot as plt
import tensorflow as tf

print("\n\n")

with tf.name_scope('CNN'):
    with tf.name_scope('Creating_estimator_function'):
        def net(features, labels, mode):
            input = tf.reshape(features['X'], [-1, 28, 28, 1])

            # receives [-1, 28, 28, 1]
            # returns [-1, 14, 14, 32]
            convolution_layer_1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5], activation=tf.nn.relu,
                                                   padding='same')
            pooling_layer_1 = tf.layers.max_pooling2d(inputs=convolution_layer_1, pool_size=[2, 2], strides=2)

            # receives [-1, 14, 14, 32]
            # returns [1, 7, 7, 64]
            convolution_layer_2 = tf.layers.conv2d(inputs=pooling_layer_1, filters=64, kernel_size=[5, 5],
                                                   activation=tf.nn.relu, padding='same')
            pooling_layer_2 = tf.layers.max_pooling2d(inputs=convolution_layer_2, pool_size=[2, 2], strides=2)

            flattening = tf.reshape(pooling_layer_2, shape=[-1, 7 * 7 * 64])

            dense_layer = tf.layers.dense(inputs=flattening, units=1024, activation=tf.nn.relu)

            dropout = tf.layers.dropout(inputs=dense_layer, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

            output_layer = tf.layers.dense(inputs=dropout, units=10)

            predictions = tf.argmax(input=output_layer, axis=1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            error = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_layer)

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.994, beta2=0.99999, epsilon=0.000001)
                training = optimizer.minimize(error, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=error, train_op=training)

            if mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)}
                return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops=eval_metric_ops, loss=error)

    with tf.name_scope('Create_Estimator'):
        classifier = tf.estimator.Estimator(model_fn=net)

    with tf.name_scope('Create_functions'):
        training_function = tf.estimator.inputs.numpy_input_fn(x={'X': X_train}, y=y_train, batch_size=128, num_epochs=None, shuffle=True)
        testing_function = tf.estimator.inputs.numpy_input_fn(x={'X': X_test}, y=y_test, num_epochs=1, shuffle=False)
        pred_function = tf.estimator.inputs.numpy_input_fn(x={'X': X_image_test}, shuffle=False)

    with tf.name_scope('Training'):
        classifier.train(input_fn=training_function, steps=150)

    with tf.name_scope('Evaluating'):
        result = classifier.evaluate(input_fn=testing_function)

        print()
        print(result)
        print()

    with tf.name_scope('Predicting'):
        pred = list(classifier.predict(input_fn=pred_function))

        print()
        print(pred)
        print()

plt.imshow(X_image_test.reshape((28, 28)), cmap='gray')
plt.show()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('C:\Graphs\CNN Graph', sess.graph)
    writer.close()