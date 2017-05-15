import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


# data import and conversion
# import csv file and create list with lines
ln = []
with open('./data_sim/driving_log.csv') as logfile:
    reader = csv.reader(logfile)
    for i in reader:
        ln.append(i)

# skip head line
ln = ln[1:]
train_samples, validation_samples = train_test_split(ln, test_size=0.2)

# generator
steer_offset = .2

cdir = os.getcwd()


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = cdir + '\\data_sim\\IMG\\' + batch_sample[0].split('\\')[-1]
                name_left = cdir + '\\data_sim\\IMG\\' + batch_sample[1].split('\\')[-1]
                name_right = cdir + '\\data_sim\\IMG\\' + batch_sample[2].split('\\')[-1]
                center_image, left_image, right_image = [cv2.imread(name), cv2.imread(name_left), cv2.imread(name_right)]
                center_image_rs = cv2.resize(center_image, (0, 0), fx=0.5, fy=0.5)
                left_image_rs = cv2.resize(left_image, (0, 0), fx=0.5, fy=0.5)
                right_image_rs = cv2.resize(right_image, (0, 0), fx=0.5, fy=0.5)
                center_angle, left_angle, right_angle = [float(batch_sample[3]), float(batch_sample[3])+steer_offset,
                                                         float(batch_sample[3])-steer_offset]
                images.append(center_image_rs)
                images.append(left_image_rs)
                images.append(right_image_rs)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

                # flipped images
                images.append(cv2.flip(center_image_rs, 1))
                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function

#define batch size
b_size = 128

train_generator = generator(train_samples, batch_size=b_size)
validation_generator = generator(validation_samples, batch_size=b_size)


# model building and training

model = Sequential()
model.add(Cropping2D(cropping=((25, 10), (0, 0)), input_shape=(80, 160, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Conv2D(12, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(36, (3, 3), activation='relu'))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.summary()

model.load_weights('weights.h5', by_name=True)

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/b_size,
                                     validation_data=validation_generator, validation_steps=len(validation_samples)/b_size,
                                     epochs=5, verbose=1)

model.save('model.h5')
model.save_weights('weights.h5')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
