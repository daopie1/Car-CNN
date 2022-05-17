# Car-CNN
Abstract

I intend to work on creating an image classification convolutional neural network for determining the make and model of a car when shown its image. I intend to train my CNN using a Stanford dataset with over 15000 images of cars (found here: https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

Introduction

Vehicle detection and identification is critical for enforcing traffic control, statistical analysis, and supporting law enforcement. When identifying a vehicle there are several different ways we can categorize, from the general like trucks and cars to the detailed like make and model. In this implementation of a vehicle detection and identification convolutional neural network we need to know the make, model, and year of manufactory, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe. We will use the Stanford dataset, a dataset that contains “16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images.”(jkrause, 2013) The goal of this project is to create a CNN model that correctly identifies a vehicle with a relatively high accuracy. To create such a model, I started with a basic CNN that only trained on 5 epochs and used a poor preprocessing algorithm. To achieve a higher accuracy, I will implement better CNN models, train using more epochs, and use a better preprocessing algorithm.

Methodology

I have begun the project by creating a basic CNN model using the Stanford dataset. In order to achieve better accuracy, I will upgrade my CNN model to three different implementations and compare their results; ResNet50, Fast-CNN, and YOLO. Evaluating the different CNN models against each other to find any competitive advantage may produce a model superior to the others. There are two other aspects that I will enhance to achieve better accuracy. This includes upgrading the preprocessing unit and using more epochs. The current preprocessing unit just resizes the dataset to a uniform 200 by 200. This cuts out potential significant data from each image. Focusing the images on the cars and resizing around them would theoretically lead to a higher accuracy rate. Using more epochs requires better hardware as my current implementation takes 20 + minutes for 5 epochs. I plan to use Google Colab to achieve this feet. However, Google Colabs tensorflow_dataset database fails to load the dataset because of a broken url on their end. Thus, I will need to implement my own data pipeline with the download images. Finally, once I have achieved an optimal CNN model, I will implement Live Vehicle Detection and Identification. 

Plan for Evaluation / Demonstration

My plan for evaluation includes measuring the accuracy and loss metrics from training, validation, and testing. Using this data, I will compare my CNN models. Achieving an accuracy of over 80% would be ideal. My first plan for demonstration includes running an image of a car through the model and judge the label given to the image by the model. My second plan for demonstration includes judging the live vehicle image identification algorithm if it outputs the correct label.
