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

def letters_extract(image_file: str, out_size=45):
    img = cv2.imread(image_file)
    img = cv2.bitwise_not(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        #if hierarchy[0][idx][3] == 0:
        cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
        letter_crop = rgb[y:y + h, x:x + w]
        # print(letter_crop.shape)

        # Resize letter canvas to square
        size_max = max(w, h)
        letter_square = 0 * np.ones(shape=[size_max, size_max, 3], dtype=np.uint8)
        if w > h:
            # Enlarge image top-bottom
            # ------
            # ======
            # ------
            y_pos = size_max//2 - h//2
            letter_square[y_pos:y_pos + h, 0:w] = letter_crop
        elif w < h:
            # Enlarge image left-right
            # --||--
            x_pos = size_max//2 - w//2
            letter_square[0:h, x_pos:x_pos + w] = letter_crop
        else:
            letter_square = letter_crop

        # Resize letter to XxX and add letter and its X-coordinate
        letters.append((x, w, cv2.resize( cv2.bitwise_not(letter_square), (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters

def take_rect(image_file: str, step=5, kernel_size=25, out_size=45):
    img = cv2.imread(image_file)
    img = cv2.bitwise_not(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = img.copy()
    letters = []
    for x in range(1, img.shape[0] - kernel_size - 1, step):
        for y in range(1, img.shape[1] - kernel_size - 1, step):
            w = kernel_size
            h = kernel_size
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = rgb[y:y + h, x:x + w]

            # size_max = max(w, h)
            # letter_square = 0 * np.ones(shape=[size_max, size_max, 3], dtype=np.uint8)
            # if w > h:
            #     # Enlarge image top-bottom
            #     # ------
            #     # ======
            #     # ------
            #     y_pos = size_max//2 - h//2
            #     letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            # elif w < h:
            #     # Enlarge image left-right
            #     # --||--
            #     x_pos = size_max//2 - w//2
            #     letter_square[0:h, x_pos:x_pos + w] = letter_crop
            # else:
            #     letter_square = letter_crop

            letters.append((x, w, cv2.resize( cv2.bitwise_not(letter_crop), (out_size, out_size), interpolation=cv2.INTER_AREA)))
    letters.sort(key=lambda x: x[0], reverse=False)
    print('size', len(letters))

    return letters

model = tf.keras.models.load_model('./own_model_3.h5')
model.summary()

classes = []
with open('classes.inf') as file:
    while True:
        line = file.readline()
        if not line:
            break
        classes.append(line)

image_path = "./0.png"

letters = take_rect(image_path, 5, 45, 45)

# i = 1
for letter in letters:
    # cv2.imshow("test" + str(i), letter[2])
    # rgb = cv2.cvtColor(letter[2], cv2.COLOR_BGR2RGB)
    # img = np.array(rgb)
    res = model.predict(letter[2][None,:,:], verbose = 0)
    if res.max() > 3.:
        print(res.max())
        print(classes[res.argmax()])
    # i += 1
cv2.waitKey(0)