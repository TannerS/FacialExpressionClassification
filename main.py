# The modules we're going to use
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
import matplotlib

# %matplotlib inline








# load the training and target data from the data set (which is csv)
train_data = np.genfromtxt('./train_data.csv',delimiter=',')
train_target = np.genfromtxt('./train_target.csv', delimiter=',')
# load the test data also
test_data = np.genfromtxt('./test_data.csv',delimiter=',')
# here we split the training data to get some data for cross validation
train_data, val_data, train_target, val_target = train_test_split(train_data, (train_target[:, np.newaxis]), test_size=0.3, random_state=42)
# display dataset shapes before processing
print('Before pre-processing, X_train size: ', train_data.shape)
print('Before pre-processing, y_train size: ', train_target.shape)
print('Before pre-processing, X_test size: ', test_data.shape)
print('Before pre-processing, X_val size: ', val_data.shape)
print('Before pre-processing, y_val size: ', val_target.shape)
# reshape the data to match the pixles of the image (-1, = stays same, 1 = 1 channel, 48 & 48means 48x48 image)
train_data = train_data.reshape(-1,1, 48,48)
test_data = test_data.reshape(-1,1, 48,48)
val_data = val_data.reshape(-1,1, 48,48)
# break down the targets into forms of 0,1, or 2
train_target = np_utils.to_categorical(train_target, 3)
val_target = np_utils.to_categorical(val_target, 3)
# display the new shapes
print('After pre-processing, X_train size: ', train_data.shape)
print('After pre-processing, y_train size: ', train_target.shape)
print('After pre-processing, X_test size: ', test_data.shape)
print('After pre-processing, X_val size: ', val_data.shape)
print('After pre-processing, y_val size: ', val_target.shape)





# create a new model
model = Sequential()
# create a convolutional layer for 2 dimensions
# this one includes the input size ofr first layer
model.add(convolutional.Conv2D(
    filters=32,
    kernel_size=(2, 2),
    padding='same',
    strides=(1, 1),
    input_shape=train_data.shape[1:],
    activation='relu'
))
# create a convolutional layer for 2 dimensions
model.add(convolutional.Conv2D(
    filters=32,
    kernel_size=(2, 2),
    padding='same',
    strides=(1, 1),
    activation='relu'
))
# create a convolutional layer for 2 dimensions
model.add(convolutional.Conv2D(
    filters=32,
    kernel_size=(2, 2),
    padding='same',
    strides=(1, 1),
    activation='relu'
))
# create a max pooling layer for 2 dimensions
model.add(pooling.MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
))
# create a convolutional layer for 2 dimensions
model.add(convolutional.Conv2D(
    filters=64,
    kernel_size=(2, 2),
    padding='same',
    strides=(1, 1),
    activation='relu'
))
# create a convolutional layer for 2 dimensions
model.add(convolutional.Conv2D(
    filters=64,
    kernel_size=(2, 2),
    padding='same',
    strides=(1, 1),
    activation='relu'
))
# create a convolutional layer for 2 dimensions
model.add(convolutional.Conv2D(
    filters=64,
    kernel_size=(2, 2),
    padding='same',
    strides=(1, 1),
    activation='relu'
))
# create a max pooling layer for 2 dimensions
model.add(pooling.MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
))
# create a convolutional layer for 2 dimensions
model.add(convolutional.Conv2D(
    filters=128,
    kernel_size=(2, 2),
    padding='same',
    strides=(1, 1),
    activation='relu'
))
# create a convolutional layer for 2 dimensions
model.add(convolutional.Conv2D(
    filters=128,
    kernel_size=(2, 2),
    padding='same',
    strides=(1, 1),
    activation='relu'
))
# create a convolutional layer for 2 dimensions
model.add(convolutional.Conv2D(
    filters=128,
    kernel_size=(2, 2),
    padding='same',
    strides=(1, 1),
    activation='relu'
))
# create a max pooling layer for 2 dimensions
model.add(pooling.MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
))
# flatten the model's layer into a deep neuron which in turn, will be part of the fully connected feedforward network
model.add(Flatten())
# create a single deep layer with a depth of 1024 for the output space
model.add(Dense(1024))
# choosing this as the activation type
model.add(Activation('relu'))
# last layer is the softmax layer to et the probability of each class
model.add(Dense(3, activation='softmax'))







# choose adam as the optimizer
# the optmizer chooses the adaptive learnign rates which are used for the Stochastic gradient descent
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# compile (configure) the model's learning process
model.compile(
    # choose adam as optimizer
    optimizer = adam,
    # uses cross entroyphy for loss function
    loss='categorical_crossentropy',
    # A metric function is similar to an loss function, except that the results from evaluating a metric are not used when training the model.
    # https://keras.io/metrics/
    metrics=['accuracy']
)
# fit the model with the data to train ig, and data to vlaidate it,
# also define the epochs and the batch size for each epochs
model.fit(train_data, train_target, validation_data=(val_data, val_target), epochs=50, batch_size=128, verbose=2)
# model.fit(train_data, train_target, validation_split=0.3, validation_data=val_data, epochs=50, batch_size=128, verbose=2)
# get loss amount and accuracy of the validation set
loss, accuracy = model.evaluate(val_data, val_target, verbose=2)
print('test loss:', loss)
print('test accuracy', accuracy)






# Show the image of one testing example
# get random number within range of the test data
temp = np.random.randint(test_data.shape[0], size=1)
# Get its prediction
output = model.predict(test_data[temp[0]].reshape(-1,1, 48, 48))
# output prediction
print(output)
# create new figure using matplotlib
plt.figure()
# display and plot image
plt.xticks(np.arange(output.shape[1]))
plt.plot(np.arange(output.shape[1]), output.T)






# get predictions
predictions = model.predict(test_data)
# The maximum value along a given axis.
# https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.argmax.html
output = predictions.argmax(axis=1)
# open new file of my choosing
f = open('./tanner_summers_hw2_results.csv','w')
# Write category header
f.write('Id,Category\n')
# loop data and print to file the iteration and it's target for that iteration at i
for i in range(0, test_data.shape[0]):
    # write data to file
    f.write(str(i) + ',' + str(output[i]) + '\n')
# close file
f.close()





