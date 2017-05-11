# The modules we're going to use
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import numpy as np
np.random.seed(64)
import matplotlib.pyplot as plt
import matplotlib

# When you execute a code to plot with a simple SHIFT-ENTER, the plot will be shown directly under the code cell
# %matplotlib inline

train_data = np.genfromtxt('./train_data.csv',delimiter=',')
train_targets = np.genfromtxt('./train_target.csv',delimiter=',')
test_data = np.genfromtxt('./test_data.csv',delimiter=',')

print('Before pre-processing, X_train size: ', train_data.shape)
print('Before pre-processing, y_train size: ', train_targets.shape)
print('Before pre-processing, X_test size: ', test_data.shape)


# Pre-processing
# what is the -1 and 1 for?***********************************************************************
train_data = train_data.reshape(-1,1, 48,48)
test_data = test_data.reshape(-1,1, 48,48)
# ***********************************************************************
train_targets = np_utils.to_categorical(train_targets, 3)


print('After pre-processing, X_train size: ', train_data.shape)
print('After pre-processing, y_train size: ', train_targets.shape)
print('After pre-processing, X_test size: ', test_data.shape)











# Create a neural net
# The Sequential model is a linear stack of layer
model = Sequential()
# https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning
# https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
# *******************************************************************
# Define dropout rate
drop_out_rate = 0.8
# Add convolutional layer
# why wouldyou add more thna one time of convolituonal layer & pooling?*********************
# explain filters, kernel, and strides
model.add(convolutional.Conv2D(
    filters=500,
    kernel_size=(3,3),
    strides=(1, 1),
    padding='same',
    input_shape=(1,48,48),
    activation='relu',
))
#
# https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
# pool size paramtere*************************************************************8
# Add max-pooling layer
model.add(pooling.MaxPooling2D(
    pool_size=(2, 2),
    padding='same',
))

# Add 2nd convolutional layer and max-pooling layer
# 2D parametters*******************************************************888
model.add(convolutional.Conv2D(
    filters=256,
    kernel_size=(1,1),
    padding='same'
))

model.add(Activation('relu'))

model.add(pooling.MaxPooling2D(
    pool_size=(2,2),
    padding='same'
))
#
# # Add 3rd convolutional layer and max-pooling layer
model.add(convolutional.Conv2D(
    filters=128,
    kernel_size=(5,5),
    padding='same'
))

model.add(Activation('relu'))

model.add(pooling.MaxPooling2D(
    pool_size=(2,2),
    padding='same'
))
#
# Add flatten layer and fully connected layer
# explain this part***************************************************
model.add(Flatten())
# dense**************************************************************8
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(drop_out_rate))
#
# # Add one more fully connected layer with softmax activation function
model.add(Dense(3))
model.add(Activation('softmax'))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Specify an optimizer to use
# **************************************************************************
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#
# Choose loss function, optimization method, and metrics (which results to display)
model.compile(
    optimizer = adam,
    loss='categorical_crossentropy',
    #*********************************************************************************
    metrics=['accuracy']
)

# Training
# epochs************************************************************
# batch **********************************************************
model.fit(train_data,train_targets, epochs=25,batch_size=64,verbose=2)

# Testing
loss, accuracy = model.evaluate(train_data,train_targets,verbose=2)
print('test loss:', loss)
print('test accuracy', accuracy)
#
#
#

#
#
#
#
#
#
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
