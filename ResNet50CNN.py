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
  image = tf.image.resize(image, (200, 200)) # Resizing the image to 224x224 dimention

  return tf.cast(image, tf.float32)/255.0, label

def resnet_block(features, bottleneck, out_filters, training):
  """Residual block."""
  with tf.variable_scope("input"):
    original = features
    features = tf.layers.conv2d(features, bottleneck, 1, activation=None)
    features = tf.layers.batch_normalization(features, training=training)
    features = tf.nn.relu(features)

  with tf.variable_scope("bottleneck"):
    features = tf.layers.conv2d(
      features, bottleneck, 3, activation=None, padding="same")
    features = tf.layers.batch_normalization(features, training=training)
    features = tf.nn.relu(features)

  with tf.variable_scope("output"):
    features = tf.layers.conv2d(features, out_filters, 1)
    in_dims = original.shape[-1].value
    if in_dims != out_filters:
      original = tf.layers.conv2d(features, out_filters, 1, activation=None,
        name="proj")
    features += original
  return features

BATCH_SIZE = 64
ds_train = ds_train.map(normalize_img)
#ds_train = ds_train.cache()
#ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)

#datagen_train = ImageDataGenerator(
#  preprocessing_function = preprocess_input,
#  rotation_range=8, 
#  width_shift_range=0.08, 
#  height_shift_range=0.08,
#  shear_range=0.10, 
#  zoom_range=0.08,
#  channel_shift_range = 10, 
#  horizontal_flip=True,
#  fill_mode="constant")

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(200, 200,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
num_classes = 197
model = keras.Sequential([
  data_augmentation,
  #Convolution
  #Convolution

  #Convolution
  #Convolution
  #Residual

  #Convolution
  
  #Convolution
  #Convolution
  #Residual

  #Convolution
  #Convolution
  #Residual



  layers.Conv2D(17, 3, padding='same', activation='relu',kernel_regularizer = keras.regularizers.l1_l2(l1=0.001, l2=0.001)),#regularization was added to help prevent overfitting
  layers.MaxPooling2D(),
  layers.Dropout(0.1), #several dropout layers were added to help reduce overfitting
  layers.Conv2D(39, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Conv2D(87, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(87, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(87, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.3),
  layers.Flatten(),
  layers.Dense(500, activation='relu'),
  layers.Dense(num_classes)
])

print(model.summary())

#model = keras.Sequential([
#keras.Input((None,None,3)),
#  data_augmentation,
#  layers.Conv2D(8, kernel_size=(3,3), padding='same', activation = 'relu'),
#  layers.Conv2D(16, (3,3), padding='same', activation='relu'),
#  layers.MaxPooling2D(pool_size=(2,2)),
#  layers.Dropout(0.25),
#  layers.GlobalAveragePooling2D(),
#  layers.Flatten(),
#  layers.Dense(32, activation='relu'),
#  layers.Dropout(0.50),
#  layers.Dense(10),
 ##layers.Dense(10)
#])

model.compile(
  optimizer=keras.optimizers.Adam(learning_rate=0.001),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=["accuracy"],
)

history = model.fit(ds_train, epochs = 5, verbose=2)
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

# Build your input pipeline
#ds = ds.shuffle(1024).batch(1).prefetch(tf.data.AUTOTUNE)
#for example in ds.take(1):
#  image, label = example["image"], example["label"]
#cv2.imwrite("output/x_train_0.img", image)
