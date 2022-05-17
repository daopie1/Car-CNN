from cv2 import norm
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import activations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.applications.imagenet_utils import preprocess_input

(ds_train, ds_test), ds_info = tfds.load(
  "Cars196",
  split=["train","test"],
  shuffle_files=True,
  as_supervised=True,
  with_info=True,
)
#np_test = tfds.as_numpy(ds_test)
#num_test  = np_test.shape[0]
# One-hot encode the testing
#y_test  = np.zeros([num_test, 10])
#for i in range(num_test):
#	if(i < 5):
#		name = "output/test_" + str(i) + ".png"
#		cv2.imwrite(name, 255*np_test[i])
#	y_test[i, ds_info[i]] = 1

ds_test = ds_test.take(1)

for images, labels in ds_test:  # only take first element of dataset
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()

print(labels)

def normalize_img(image, label):
  # normalize images
  image = tf.image.resize(image, (200, 200)) # Resizing the image to 224x224 dimention
  #image = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15)
  return tf.cast(image, tf.float32)/255.0, label

#ds_test = tf.image.resize(ds_test, (200,200))
ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)



# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("output/model")

#cv2.imwrite("output/test.png", ds_test)

#img_array = tfds.as_numpy(ds_test)
#img_array = tf.keras.utils.img_to_array(img)
#img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = reconstructed_model.predict(ds_test)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(np.argmax(score), 100 * np.max(score))
)

cv2.imwrite("output/test.png", numpy_images)
