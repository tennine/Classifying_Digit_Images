#    name: pa2Template.py
# purpose: template for building a Keras model
#          for hand written number classification
#    NOTE: Submit a different python file for each model
# -------------------------------------------------

"""
The argparse code in the template is a very efficient and clean way to pass parameters to a python code.

In this case, the code is intended to run from the command line in the following manner:


python3 pa2Template.py --training_x MNISTXtrain1.npy --training_y MNISTytrain1.npy \
  --outModelFile nameofthemodel ...

  where nameofthemodel is the name you want to give to the h5 file.

  Note that inside the template you get the passed arguments as follows:

  parms = parseArguments()

    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)

    and later, after you built your model,

    model.save(parms.outModelFile)

 Note that in the template file, the training data gets transformed with:

 	 (X_train, y_train) = processTestData(X_train,y_train)


 You must do the same with the test data I provide before you use it to test your model.

    YOU SHOULDN'T USE AN ALTERNATIVE WAY TO DO THIS. VERY IMPORTANTLY YOU SHOULD NOT USE A
    PYTHON NOTEBOOK. IF YOU DO, THE TEST I WILL SUBJECT YOUR MODEL TO WILL FAIL AND YOU WILL LOSE 30 POINTS OF CREDIT.

    open the console from the tools menu and python console to run it
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.models import load_model
from keras import metrics

print("help")
from pa2pre1 import processTestData
import argparse


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Build a Keras model for Image classification')

    parser.add_argument('--training_x', action='store',
                        dest='XFile', default="", required=True,
                        help='matrix of training images in npy')
    parser.add_argument('--training_y', action='store',
                        dest='yFile', default="", required=True,
                        help='labels for training set')

    parser.add_argument('--outModelFile', action='store',
                        dest='outModelFile', default="", required=True,
                        help='model name for your Keras model')

    return parser.parse_args()

def main():
    np.random.seed(1671)
   # parms = parseArguments()

    #X_train = np.load(parms.XFile)
    #y_train = np.load(parms.yFile)
    X_train = np.load('MNISTXtrain1.npy')
    y_train = np.load('MNISTytrain1.npy')
    (X_train, y_train) = processTestData(X_train, y_train)

    print('KERAS modeling build starting...')
    ## Build your model here

    # made the model
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=500, activation = 'relu'))
    model.add(tf.keras.layers.Dense(units=500, activation = 'relu'))
    model.add(tf.keras.layers.Dense(units=10, activation = 'sigmoid'))

    # sets up the model
    model.compile(optimizer = keras.optimizers.RMSprop(), loss = keras.losses.CategoricalCrossentropy(), metrics = [keras.metrics.SparseCategoricalAccuracy()])
    #fits the model
    model.fit(X_train, y_train, batch_size = 64, epochs=250)

    #runs the model
    x_test = np.load('MNIST_X_test_1.npy')
    y_test = np.load('MNIST_Y_test_1.npy')
    (X_test, y_test) = processTestData(x_test, y_test)


    y_pred = model.predict(x_test)
    prediction = np.array([x_test[0], y_pred[0]])
    for y_point in range(len(y_pred)-1):
        new_prediction = np.array([x_test[y_point+1], y_pred[y_point+1]])
        prediction = np.stack([prediction, new_prediction])
    np.save("model_1_prediction", prediction)

## save your model
    model.save("m1")


if __name__ == '__main__':
    main()


'''
model = Sequential()
model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', input_shape=(x_train.shape[1], x_train.shape[2]),
               return_sequences=True))
model.add(LSTM(256, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=1)

import tensorflow as tf

x_train = tf.random.normal((2066, 300, 2))
y_train = tf.random.normal((2066, 60, 1))
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, activation='tanh', recurrent_activation='sigmoid', input_shape=(x_train.shape[1], x_train.shape[2]),
               return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(y_train.shape[1]))
model.add(tf.keras.layers.Reshape((y_train.shape[1], y_train.shape[2])))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=1, batch_size=16, validation_split=0.2, verbose=1)
Share
Improve this answer
Follow
answered Feb 16 at 7:18
'''