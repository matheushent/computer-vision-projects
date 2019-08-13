from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('mnist/', one_hot=False)

X_train = mnist.train.images
y_train = mnist.train.labels

X_test = mnist.test.images
y_test = mnist.test.labels

y_train = np.asarray(y_train, dtype=np.int32)
y_test = np.asarray(y_test, dtype=np.int32)

import matplotlib.pyplot as plt
import tensorflow as tf
tf.reset_default_graph()

# features -> pixels we have to pass
# labels -> correct answers
# mode ->

def net(features, labels, mode):
    # batch_size, width, height, channels
    input = tf.reshape(features['X'], [-1, 28, 28, 1])

    # from where the data comes, how many feature maps I want, size of feature detector
    # receives [batch_size, 28, 28, 1]
    # returns [batch_size, 28, 28, 32]
    convolution1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, padding='same')

    # strides means how many squares I want to ''jump''
    # receives [batch_size, 28, 28, 32]
    # returns [batch_size, 14, 14, 32]
    pooling1 = tf.layers.max_pooling2d(inputs=convolution1, pool_size=[2, 2], strides=2)

    # receives [batch_size, 14, 14, 32]
    # returns [batch_size, 14, 14, 64]
    convolution2 = tf.layers.conv2d(inputs=pooling1, filters=64, kernel_size=[5, 5], activation=tf.nn.relu, padding='same')

    # receives [batch_size, 14, 14, 64]
    # returns [batch_size, 7, 7, 64]
    pooling2 = tf.layers.max_pooling2d(inputs=convolution2, pool_size=[2, 2], strides=2)

    # receives [batch_size, 7, 7, 64]
    # returns [batch_size, 7 * 7 * 64]
    flattening = tf.reshape(pooling2, [-1, 7 * 7 * 64])

    # 3136 (inputs) -> 1024 (hidden) -> 10 (output)
    dense = tf.layers.dense(inputs=flattening, units=1024, activation=tf.nn.relu)

    # dropout
    dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    # receives [batch_size, 1024]
    # returns [batch_size, 10]
    output = tf.layers.dense(inputs=dropout, units=10)


    predictions = tf.argmax(output, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    error = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        training = optimizer.minimize(error, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=error, train_op=training)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels = labels, predictions = predictions)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=error, eval_metric_ops=eval_metric_ops)

classifier = tf.estimator.Estimator(model_fn=net)

# Args:

# features: This is the first item returned from the input_fn passed to train, evaluate, and predict. This should be a single tf.Tensor or dict of same.
# labels: This is the second item returned from the input_fn passed to train, evaluate, and predict. This should be a single tf.Tensor or dict of same (for multi-head models). If mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed. If the model_fn's signature does not accept mode, the model_fn must still be able to handle labels=None.
# mode: Optional. Specifies if this training, evaluation or prediction. See tf.estimator.ModeKeys.
# params: Optional dict of hyperparameters. Will receive what is passed to Estimator in params parameter. This allows to configure Estimators from hyper parameter tuning.
# config: Optional estimator.RunConfig object. Will receive what is passed to Estimator as its config parameter, or a default value. Allows setting up things in your model_fn based on configuration such as num_ps_replicas, or model_dir.

training_function = tf.estimator.inputs.numpy_input_fn(x = {'X': X_train}, y = y_train, batch_size=128, num_epochs=None, shuffle=True)

classifier.train(input_fn=training_function, steps=50)    

testing_function = tf.estimator.inputs.numpy_input_fn(x = {'X': X_test}, y = y_test, num_epochs=1, shuffle=False)

result = classifier.evaluate(input_fn=testing_function)
print("\n\n")
print(result)

print("\n\n")

"""X_imagem_teste = X_test[0].reshape(-1, 1)

pred_function = tf.estimator.inputs.numpy_input_fn(x = {'X': X_imagem_teste}, shuffle=False)
pred = list(classifier.predict(input_fn=pred_function))

print(pred)
print("\n\n")

plt.imshow(X_imagem_teste.reshape((28, 28)), cmap='gray')
plt.show()"""