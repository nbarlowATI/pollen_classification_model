import os
import requests
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
# prevent annoying tensorflow warning

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import warnings
warnings.simplefilter("ignore")

CLASS_LABELS = ['anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena', 'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis', 'mabea', 'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus', 'senegalia', 'serjania', 'syagrus', 'tridax', 'urochloa']

def load_model_with_weights(url):
    ### TODO ####
    ### Need to download the weights file from url, if it's not already
    ### present, and put the downloaded filename into a variable
    ### called weights_filename
    model = tf.keras.models.load_model(weights_filename)
    return model

def preprocess_image(image):
    """
    Ensure that an input image is the correct size, and
    has the expected shape, to be used by the predict function

    parameters
    ----------
    image: np.ndarray, shape(npix_x,npix_y,3)

    returns
    -------
    image: np.ndarray, shape(None, 224, 224, 3)
    """
    image = resize(image, (224, 224),
                   preserve_range=True,
                   anti_aliasing=True)
    image = np.expand_dims(image, 0)
    return image


class efficientNetB3:
    ### TODO ####
    ### Add a constructor to this class that calls the function
    ### to download the model weights, load the model, and assign
    ### to self.model

    def predict(self, image: np.ndarray):
        ### TODO - make sure the image is the correct size, and has
        ### the dimensions expected by the model.

        result = self.model.predict(image)
        ### TODO ####
        ### Find the highest weight, and, using the list of CLASS_LABELS
        ### get the corresponding class name.
        return "FIXME"



if __name__ == "__main__":
    pass
