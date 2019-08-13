from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = load_model('C:/Users/mathe/Desktop/CNN-GAN/CNN/CIFAR-10/logs/model.h5')

img_path = 'C:/Users/mathe/Desktop/CNN-GAN/CNN/CIFAR-10/assets/airplane.png'

img = image.load_img(img_path)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

pred = model.predict(x)

index = np.argmax(pred[0])

# Output the feature map of the conv2d_4, the last
# convolutional layer in our model
airplane_output = model.output[:, index]
last_conv_layer = model.get_layer('conv2d_4')

# Gradients of the airplane class wrt the conv2d_4 filter
grads = K.gradients(airplane_output, last_conv_layer.output)[0]

# Each entry is the mean intensity of the gradient over a specific feature-map channel 
pooled_grads = K.mean(grads, axis=[0, 1, 2])
print(pooled_grads)

# Accesses the values we just defined given our sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# Values of pooled_grads_value, conv_layer_output_value given our input image
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature-map array by the 'importance' 
# of this channel regarding the input image 
for i in range(64):
    #channel-wise mean of the resulting feature map is the Heatmap of the CAM
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

## Ploting out Heatmap
#heatmap = np.maximum(heatmap, 0)
#heatmap /= np.max(heatmap)
#plt.matshow(heatmap)
#plt.show()