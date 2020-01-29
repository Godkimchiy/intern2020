import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.models import load_model
model = load_model('MNIST_0129_model.h5')

img_four = plt.imread('C:/Users/smkim/Desktop/chiyoon/number4.png')
test_num = cv2.resize(img_four, (28,28))[:,:,1]
test_num = (test_num < 70)*test_num
test_num = test_num.astype('float32')/255.

plt.imshow(test_num, cmap='Greys', interpolation='nearest')
test_num = test_num.reshape((1,28,28,1))

print('The Answer is ', model.predict_classes(test_num))