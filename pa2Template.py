#    name: pa2Template.py
# purpose: template for building a Keras model
#          for hand written number classification
#    NOTE: Submit a different python file for each model
# -------------------------------------------------

'''
The argparse code in the template is a very efficient and clean way to pass parameters to a python code.

In this case, the code is intended to run from the command line in the following manner:

python3 pa2Template.py --training_x MNISTXtrain.npy --training_y MNISTytrain.npy \
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
'''





import numpy as np
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model

from pa2pre import processTestData
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

    parms = parseArguments()

    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)

    (X_train, y_train) = processTestData(X_train,y_train)

    print('KERAS modeling build starting...')
    ## Build your model here

    ## save your model


if __name__ == '__main__':
    main()
    