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

def res_identity(x, f1, f2): 
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block 

    #first block keras.regularizers.l1_l2(l1=0.001, l2=0.001)
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(l2=0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    #second block # bottleneck (but size kept same with padding)
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(l2=0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(l2=0.001))(x)
    x = layers.BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input 
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activations.relu)(x)

    return x

def res_conv(x,s,f1,f2):
    x_skip = x

    # first block
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(l2=0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    # second block
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(l2=0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    #third block
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(l2=0.001))(x)
    x = layers.BatchNormalization()(x)

    # shortcut 
    x_skip = layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(l2=0.001))(x_skip)
    x_skip = layers.BatchNormalization()(x_skip)

    # add 
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activations.relu)(x)

    return x

def resnet50():

    input_im = layers.Input(shape=(200,200,3))

    pre_im = layers.experimental.preprocessing.RandomFlip("horizontal")(input_im)
    pre_im = layers.experimental.preprocessing.RandomRotation(0.1)(pre_im)
    pre_im = layers.experimental.preprocessing.RandomZoom(0.1)(pre_im)

    x = layers.ZeroPadding2D(padding=(3, 3))(pre_im)

    # 1st stage
    # here we perform maxpooling, see the figure above

    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    #2nd stage 
    # frm here on only conv block and identity block, no pooling

    x = res_conv(x, s=1, f1=64, f2=256)
    x = res_identity(x, f1=64, f2=256)
    x = res_identity(x, f1=64, f2=256)

    # 3rd stage

    x = res_conv(x, s=2, f1=128, f2=512)
    x = res_identity(x, f1=128, f2=512)
    x = res_identity(x, f1=128, f2=512)
    x = res_identity(x, f1=128, f2=512)

    # 4th stage

    x = res_conv(x, s=2, f1=256, f2=1024)
    x = res_identity(x, f1=256, f2=1024)
    x = res_identity(x, f1=256, f2=1024)
    x = res_identity(x, f1=256, f2=1024)
    x = res_identity(x, f1=256, f2=1024)
    x = res_identity(x, f1=256, f2=1024)

    # 5th stage

    x = res_conv(x, s=2, f1=512, f2=2048)
    x = res_identity(x, f1=512, f2=2048)
    x = res_identity(x, f1=512, f2=2048)

    # ends with average pooling and dense connection

    x = layers.GlobalAveragePooling2D((2, 2), padding='same')(x)

    #x = layers.Flatten()(x)
    x = layers.Dense(197, activation='softmax', kernel_initializer='he_normal')(x) #multi-class #num_classes = 197

    # define the model 

    model = tf.keras.Model(inputs=input_im, outputs=x, name='Resnet50')

    return model

BATCH_SIZE = 64
ds_train = ds_train.map(normalize_img)
#ds_train = ds_train.cache()
#ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)

model = resnet50()
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

#model.compile(
#  optimizer=keras.optimizers.Adam(learning_rate=0.001),
#  loss=keras.losses.CategoricalCrossentropy(),
#  metrics=["accuracy"],
#)

#history = model.fit(ds_train, epochs = 5, verbose=2)
#model.evaluate(ds_test)

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