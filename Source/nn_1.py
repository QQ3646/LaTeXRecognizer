import fnmatch
import os

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

def create_model():
    model = Sequential()
    model.add(tf.keras.layers.Rescaling(1./255))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(45, 45, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(82))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    return model

# letters = letters_extract("./0.png")

# i = 1
# for letter in letters:
#     cv2.imshow("test" + str(i), letter[2])
#     i += 1
# cv2.waitKey(0)

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
# model = create_model()
model = tf.keras.models.load_model('./own_model_1.h5')
model.summary()

# x_train = []
# y_train = []

# for root, dirnames, filenames in os.walk('../extracted_images'):
#     i = 0
#     for dir in dirnames:
#         print("Current i:" + str(i))
#         if i < 2:
#             for aroot, adirnames, afilenames in os.walk(root + '/' + dir):
#                 for filename in fnmatch.filter(afilenames, '*.jpg'):
#                     matches = [0.]*82
#                     matches[i] = 1.
#                     y_train.append(matches)
#                     current_path_to_file = root + '/' + dir + '/' + filename
#                     # print(current_path_to_file)
#                     # img = cv2.imread(current_path_to_file)
#                     # inp  = cv2.resize(img, (45, 45))
#                     # rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
#                     # rgb = (rgb[...,::-1].astype(np.float32)) / 255.0
#                     # rgb = rgb.reshape(1, 45*45*3)
#                     # rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
#                     # rgb_tensor = tf.expand_dims(rgb_tensor , 0)
#                     # x_train.append(np.array(rgb))
#             i += 1

root = '../extracted_images/train'

train_ds = tf.keras.utils.image_dataset_from_directory(
    root,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(45, 45),
    batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(
    root,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(45, 45),
    batch_size=32)

class_names = train_ds.class_names
print(class_names)

file = open("./classes.inf", "w")
for class_name in class_names:
    file.write(class_name)
    file.write('\n')

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Ready!")
input()

model.fit(train_ds, validation_data=val_ds, epochs=3)

# x_train = x_train.astype('float32')
# y_train = y_train.astype('float32')

# x_val = x_train[-10000:]
# y_val = y_train[-10000:]
# x_train = x_train[:-10000]
# y_train = y_train[:-10000]

# print(len(x_train))
# print(len(y_train))

# print("Ready!")
# input()

# model.fit(x_train, y_train, validation_data=(x_train, y_train), callbacks=[learning_rate_reduction], epochs=30)
model.save('own_model.h5')