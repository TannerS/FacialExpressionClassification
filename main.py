# The modules we're going to use
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from sklearn.model_selection import StratifiedKFold
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


# http://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
# https://datascience.stackexchange.com/questions/11747/cross-validation-in-keras
# skf = StratifiedKFold(n_splits=10,random_state=64, shuffle=True)
#
# cvscores = []
#
# for train_index, test_index in skf.split, train_targets):

model = Sequential()

#
model.add(convolutional.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    padding='same',
    strides=(1, 1),
    input_shape=train_data.shape[1:],
    activation='relu'
))


# model.add(convolutional.Conv2D(
#     filters=16,
#     kernel_size=(3, 3),
#     padding='same',
#     strides=(1, 1),
#     activation='relu'
# ))

model.add(convolutional.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding='same',
    strides=(1, 1),
    activation='relu',
))

# model.add(Dropout(0.1))

model.add(pooling.MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
))



model.add(convolutional.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding='same',
    strides=(1, 1),
    input_shape=train_data.shape[1:],
    activation='relu'
))

# model.add(convolutional.Conv2D(
#     filters=32,
#     kernel_size=(3, 3),
#     padding='same',
#     strides=(1, 1),
#     activation='relu'
# ))

model.add(convolutional.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    padding='same',
    strides=(1, 1),
    activation='relu',
))

# model.add(Dropout(0.1))

model.add(pooling.MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
))


model.add(convolutional.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    padding='same',
    strides=(1, 1),
    activation='relu'
))

# model.add(convolutional.Conv2D(
#     filters=64,
#     kernel_size=(3, 3),
#     padding='same',
#     strides=(1, 1),
#     activation='relu'
# ))

model.add(convolutional.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    padding='same',
    strides=(1, 1),
    activation='relu',
))

# model.add(Dropout(0.1))

model.add(pooling.MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
))











#
# #
# model.add(convolutional.Conv2D(
#     filters=128,
#     kernel_size=(3, 3),
#     padding='same',
#     strides=(1, 1),
#     activation='relu'
# ))
#
# model.add(convolutional.Conv2D(
#     filters=128,
#     kernel_size=(3, 3),
#     padding='same',
#     strides=(2, 2),
#     activation='relu'
# ))
#
# model.add(convolutional.Conv2D(
#     filters=128,
#     kernel_size=(3, 3),
#     padding='same',
#     strides=(3, 3),
#     activation='relu',
# ))
#
# model.add(pooling.MaxPooling2D(
#     pool_size=(2, 2),
#     padding='same',
# ))
#
# model.add(Dropout(0.5))
#
# model.add(pooling.MaxPooling2D(
#     pool_size=(2, 2),
#     padding='same',
# ))



model.add(Flatten())

# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# # model.add(Dropout(0.8))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))



adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(
    optimizer = adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model.fit(train_data, train_targets,epochs=10,batch_size=32,verbose=2)
model.fit(train_data, train_targets, validation_split=0.3, validation_data=test_data.all(), epochs=50, batch_size=64, verbose=2)
# model.fit(train_data[train_index], train_targets[test_index], epochs=10, batch_size=64, verbose=2)
# scores = model.evaluate(train_data[test_index], train_targets[test_index], verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# model.
# loss, accuracy = model.evaluate(test_data, test_target, verbose=2)
# print('test loss:', loss)
# print('test accuracy', accuracy)


# output = model.predict(X_test[temp[0]].reshape(-1,1, 28,28))


# http://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/




predictions = model.predict(test_data)

output = predictions.argmax(axis=1)

f = open('./result3.csv','w')
f.write('Id,Category\n')
for i in range(0, test_data.shape[0]):
    f.write(str(i) + ',' + str(output[i]) + '\n')
f.close()











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
