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
import seaborn as sns
import matplotlib.pylab as plt
from pa2pre4 import processTestData
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

    # X_train = np.load(parms.XFile)
    # y_train = np.load(parms.yFile)
    X_train = np.load('MNISTXtrain1.npy')
    y_train = np.load('MNISTytrain1.npy')
    (X_train, y_train) = processTestData(X_train, y_train)

    print('KERAS modeling build starting...')
    ## Build your model here

    # made the model
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=500, activation='relu'))
    model.add(tf.keras.layers.Dense(units=500, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

    # sets up the model
    model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.CategoricalAccuracy()])
    # fits the model
    model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=1, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=2)])

    # runs the model
    x_test = np.load('MNIST_X_test_1.npy')
    y_test = np.load('MNIST_Y_test_1.npy')
    (x_test, y_test) = processTestData(x_test, y_test)
    predict_y = model.predict(x_test)
    true_class = tf.argmax(y_test,1)
    predicted_class = tf.argmax(predict_y,1)
    matrix = tf.math.confusion_matrix(true_class, predicted_class, num_classes=10)
    ax = sns.heatmap(matrix, annot=True, fmt='.1f',linewidth=0.5)
    plt.xlabel('True Class')
    plt.ylabel('Predicted Class')
    plt.show()
    #np.save('h1_predict_y',predict_y)

    #sns.set()
    ax = sns.heatmap(predict_y, annot=True, fmt='.1f', linewidth=0.5)
    #plt.xlabel('True Class')
    #plt.ylabel('Predicted Class')
    #plt.show()

    ## save your model
  #  model.save("m1")


if __name__ == '__main__':
    main()
