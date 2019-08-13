import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.reset_default_graph()

mnist = input_data.read_data_sets('mnist/', one_hot=True)

# Test
plt.imshow(mnist.train.images[0].reshape(28, 28), cmap='gray')
# plt.show()

# Simulation

image1 = np.arange(0, 784).reshape(28, 28)
plt.imshow(image1)
# plt.show()

image2 = np.random.normal(size=784).reshape(28, 28)
plt.imshow(image2)
# plt.show()

ph_noise = tf.placeholder(tf.float32, [None, 100])

def generator(noise, reuse=None):
    with tf.VariableScope('generator', reuse=reuse):
        # 100 -> 128 -> 128 -> 784

        hidden_layer1 = tf.nn.relu(tf.layers.dense(inputs=noise, units=128))
        hidden_layer2 = tf.nn.relu(tf.layers.dense(inputs=hidden_layer1, units=128))
        hidden_layer3 = tf.layers.dense(inputs=hidden_layer2, units=784, activation=tf.nn.tanh)
        return hidden_layer3

real_image_ph = tf.placeholder(tf.float32, [None, 784])

def discriminator(X, reuse=None):
    with tf.VariableScope('discriminator', reuse=reuse):
        # 784 -> 128 -> 128 -> 1

        hidden_layer1 = tf.nn.relu(tf.layers.dense(inputs=X, units=128))
        hidden_layer2 = tf.nn.relu(tf.layers.dense(inputs=hidden_layer1, units=128))
        logits = tf.layers.dense(hidden_layer2, units=1)
        return logits

logits_real_images = discriminator(real_image_ph)
logits_noise_images = discriminator(generator(ph_noise), reuse=True)

real_discriminator_error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real_images, labels=tf.ones_like(logits_real_images) * (0.9)))
noise_discriminator_error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_noise_images, labels=tf.zeros_like(logits_noise_images)))
discriminator_error = real_discriminator_error + noise_discriminator_error
generator_error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_noise_images, labels=tf.ones_like(logits_noise_images)))

variables = tf.trainable_variables() # To see the variables just use print(variables)
discriminator_variables = [v for v in variables if 'discriminator' in v.name]
generator_variables = [v for v in variables if 'generator' in v.name]

discriminator_training = tf.train.AdamOptimizer(learning_rate=0.001).minimize(discriminator_error, var_list=discriminator_variables)
generator_training = tf.train.AdamOptimizer(learning_rate=0.001).minimize(generator_error, var_list=generator_variables)


batch_size = 100
test_samples = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(300):
        batch_num = mnist.train.num_examples // batch_size
        for i in range(batch_num):
            batch = mnist.train.next_batch(100)
            image_batch = batch[0].reshape((100, 784))
            image_batch = image_batch * 2 - 1

            noise_batch = np.random.uniform(-1, 1, size=(batch_size, 100))

            _, discriminator_cost = sess.run([discriminator_training, discriminator_error], feed_dict={real_image_ph: image_batch, ph_noise: noise_batch})

            _, generator_cost = sess.run([generator_training, generator_error], feed_dict={ph_noise: noise_batch})

        print('Epoch: ' + str(epoch + 1) + ' D error: ' + str(discriminator_cost) + ' G error: ' + str(generator_error))

        noise_test = np.random.uniform(-1, 1, size=(1, 100))
        generated_image = sess.run(generator(ph_noise, reuse=True), feed_dict={ph_noise: noise_test})
        test_samples.append(generated_image)

plt.imshow(test_samples[0].reshape(28, 28), cmap='Greys')
plt.show()

plt.imshow(test_samples[10].reshape(28, 28), cmap='Greys')
plt.show()

plt.imshow(test_samples[30].reshape(28, 28), cmap='Greys')
plt.show()

plt.imshow(test_samples[49].reshape(28, 28), cmap='Greys')
plt.show()

plt.imshow(test_samples[100].reshape(28, 28), cmap='Greys')
plt.show()

plt.imshow(test_samples[250].reshape(28, 28), cmap='Greys')
plt.show()

plt.imshow(test_samples[299].reshape(28, 28), cmap='Greys')
plt.show()