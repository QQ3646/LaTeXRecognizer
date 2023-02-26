import cv2
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.constraints import maxnorm
import tensorflow as tf

model = tf.keras.models.load_model('./own_model_1.h5')
model.summary()

image_path = '../extracted_images/predict/2.jpg'

classes = []
with open('classes.inf') as file:
    while True:
        line = file.readline()
        if not line:
            break
        classes.append(line)

img = tf.keras.utils.load_img(image_path, color_mode='rgb', target_size=(45, 45))
img = np.array(img)

res = model.predict(img[None,:,:])

print(res)
print(res.shape)
print(res.argmax())
print(classes[res.argmax()])