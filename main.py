# The modules we're going to use
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
import matplotlib


train_data = np.genfromtxt('./train_data.csv',delimiter=',')
train_targets = np.genfromtxt('./train_target.csv',delimiter=',')
test_data = np.genfromtxt('./test_data.csv',delimiter=',')

print('Before pre-processing, X_train size: ', train_data.shape)
print('Before pre-processing, y_train size: ', train_targets.shape)
print('Before pre-processing, X_test size: ', test_data.shape)

train_data = train_data.reshape(-1,1, 48,48)
test_data = test_data.reshape(-1,1, 48,48)

train_targets = np_utils.to_categorical(train_targets, 3)

print('After pre-processing, X_train size: ', train_data.shape)
print('After pre-processing, y_train size: ', train_targets.shape)
print('After pre-processing, X_test size: ', test_data.shape)


model = Sequential()
drop_out_rate = 0.5

############### START FIRST CONVOLUTIONAL LAYER (3) + MAX POOLING (1) ###############


model.add(convolutional.Conv2D(
    filters=16,
    kernel_size=(2,2),
    strides=(1, 1),
    padding='same',
    input_shape=(1,48,48),
    activation='relu',
))


model.add(convolutional.Conv2D(
    filters=16,
    kernel_size=(2,2),
    strides=(1, 1),
    padding='same',
    # input_shape=(1,48,48),
    # activation='relu',
))

model.add(convolutional.Conv2D(
    filters=16,
    kernel_size=(2,2),
    strides=(1, 1),
    padding='same',
    # input_shape=(1,48,48),
    # activation='relu',
))

# model.add(Activation('relu'))

model.add(pooling.MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
))



############### START SECOND CONVOLUTIONAL LAYER (3) + MAX POOLING (1) ###############


model.add(convolutional.Conv2D(
    filters=16,
    kernel_size=(2,2),
    strides=(1, 1),
    padding='same',
    # input_shape=(1,48,48),
    # activation='relu',
))


model.add(convolutional.Conv2D(
    filters=16,
    kernel_size=(2,2),
    strides=(1, 1),
    padding='same',
    # input_shape=(1,48,48),
    activation='relu',
))

model.add(convolutional.Conv2D(
    filters=16,
    kernel_size=(2,2),
    strides=(1, 1),
    padding='same',
    # input_shape=(1,48,48),
    # activation='relu',
))

# model.add(Activation('relu'))

model.add(pooling.MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
))


############### START THIRD CONVOLUTIONAL LAYER (3) + MAX POOLING (1) ###############

model.add(convolutional.Conv2D(
    filters=16,
    kernel_size=(2,2),
    strides=(1, 1),
    padding='same',
    # input_shape=(1,48,48),
    # activation='relu',
))


model.add(convolutional.Conv2D(
    filters=16,
    kernel_size=(2,2),
    strides=(1, 1),
    padding='same',
    # input_shape=(1,48,48),
    # activation='relu',
))

model.add(convolutional.Conv2D(
    filters=16,
    kernel_size=(2,2),
    strides=(1, 1),
    padding='same',
    # input_shape=(1,48,48),
    # activation='relu',
))

# model.add(Activation('relu'))

model.add(pooling.MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
))

############### END CONVOLUTIONAL LAYER (3) + MAX POOLING (1) ###############


model.add(Flatten())

############### START FIRST FULLY CONNECTED LAYER ###############

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(drop_out_rate))

############### SECOND FIRST FULLY CONNECTED LAYER ###############

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(drop_out_rate))

############### THIRD FIRST FULLY CONNECTED LAYER ###############

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(drop_out_rate))

############### FOURTH FIRST FULLY CONNECTED LAYER ###############

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(drop_out_rate))

############### FOURTH FIRST FULLY CONNECTED LAYER ###############

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(drop_out_rate))

############### FOURTH FIRST FULLY CONNECTED LAYER ###############

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(drop_out_rate))

############### FIFTH FIRST FULLY CONNECTED LAYER ###############

model.add(Dense(3))
model.add(Activation('softmax'))

############### END FULLY CONNECTED LAYER ###############

# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(
    # optimizer = adam,
    optimizer = 'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model.fit(train_data, train_targets,epochs=10,batch_size=32,verbose=2)
model.fit(train_data, train_targets, validation_split=0.20, epochs=30, batch_size=128, verbose=2)

# loss, accuracy = model.evaluate(test_data, test_target, verbose=2)
# print('test loss:', loss)
# print('test accuracy', accuracy)


# output = model.predict(X_test[temp[0]].reshape(-1,1, 28,28))


# http://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/















#
# # Show the image of one testing example
# temp = np.random.randint(y_test.shape[0],size=1)
# plt.imshow(X_test[temp[0],0,:,:],cmap='gray')
#
# # Display its target
# print(y_test[temp[0]])
#
# # Get its prediction
# output = model.predict(X_test[temp[0]].reshape(-1,1, 28,28))

# print(output)
# plt.figure()
# plt.xticks(np.arange(output.shape[1]))
# plt.plot(np.arange(output.shape[1]),output.T)
#
#
#
