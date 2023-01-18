from re import I
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
from tensorflow.keras.applications import ResNet50


np.set_printoptions(suppress=True)   # suppress scientific notation

# Do I need a check for if a folder already exists?
os.system("mkdir output")


#print("HI")
# Construct a tf.data.Dataset
#ds = tfds.load('Cars196', split='train', shuffle_files=True)
(ds_train, ds_test), ds_info = tfds.load(
  "Cars196",
  split=["train","test"],
  shuffle_files=True,
  as_supervised=True,
  with_info=True,
)

#fig = tfds.show_examples(ds_train, ds_info, rows = 4, cols = 4) #as_supervised=False
#print(ds_info)

def normalize_img(image, label):
    # normalize images
    image = tf.image.resize(image, (224, 224)) # Resizing the image to 224x224 dimention

    return tf.cast(image, tf.float32)/255.0, label


BATCH_SIZE = 64
ds_train = ds_train.map(normalize_img)
#ds_train = ds_train.cache()
#ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)

inputs = layers.Input(shape=(224,224,3))

base_model = ResNet50(include_top=True, input_tensor = inputs)
last_layer = base_model.layers[-2].output 
out = layers.Dense(units = 197, activation = 'softmax', name = 'output')(last_layer)
model = keras.Model(inputs = inputs, outputs = out)
for layer in model.layers[:-25]:
  layer.trainable = False

print(model.summary())

model.compile(
  optimizer=keras.optimizers.Adam(learning_rate=0.001),
  loss=keras.losses.SparseCategoricalCrossentropy(),
  metrics=["accuracy"],
)

#callback = tf.keras.callbacks.EarlyStopping(monitor='loss',mode = 'min')
history = model.fit(ds_train, epochs=50)

#history = model.fit(ds_train, epochs = 5, verbose=2)
model.evaluate(ds_test)

#the code below to plot accuracy and loss curves were used in Google's tutorial
acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
loss = history.history['loss']
#val_loss = history.history['val_loss']
epochs_range = range(len(history.epoch))
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
name = "output/accuracy.png"
plt.savefig(name)
plt.clf()

model.save("output/model")