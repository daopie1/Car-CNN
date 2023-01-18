Check the most recent release!
The Final Report and relevant files are there.

# Car-CNN
Abstract

I present image classification convolutional neural network for determining the 
make and model of a car when shown its image. I have created trained CNN 
models with high accuracy using a Stanford dataset with over 15000 images of 
cars (found here: https://ai.stanford.edu/~jkrause/cars/car_dataset.html). Using 
those models, I attempt to accurately predict the car make, model, and year from 
an image.

Introduction

Vehicle detection and identification is critical for enforcing traffic control, statistical analysis, and 
supporting law enforcement. When identifying a vehicle there are several different ways we can 
categorize, from the general like trucks and cars to the detailed like make and model. In this 
implementation of a vehicle detection and identification convolutional neural network we need to 
know the make, model, and year of manufactory, e.g. 2012 Tesla Model S or 2012 BMW M3 
coupe. We use the Stanford dataset, a dataset that contains “16,185 images of 196 classes of cars. 
The data is split into 8,144 training images and 8,041 testing images.”(Krause, 2013) The goal of 
this project is to create a CNN model that correctly identifies a vehicle with a relatively high 
accuracy. To create such a model, I started with a basic CNN that only trained on 5 epochs. To 
achieve a higher accuracy, I implemented a ResNet50 CNN framework and trained using more 
epochs.

Related Work

The beginning of vehicle detection and classification seems to have started almost 10 years ago 
with 3D Object representations for fine-grained categorization (Krause et al. 2013). Over those 10 
years the machine learning community and technology as a whole has made leaps and bounds in 
advancements. Even so the methodology behind their work still stands and you can see their dataset 
is one of many standards used for training today. Those achievements made in between then and 
now have given better access to stronger computations and stronger neural networks.
  Liu et. al (2015) created a baseline for Image classification of vehicle make and model and 
are what I base my project on. However, their CNN models have become somewhat outdated and 
current models can achieve much better results.
  Maity et. al (2021) builds the foundation for my future attempt with object detection using 
Faster R-CNN and YOLO based Vehicle detection.
  Bautista et. al (2016) on Convolutional neural network for vehicle detection in low 
resolution traffic videos does not attempt to identify vehicles by make and model like my 
implementation will.
  Potdar et. al (2018) on A Convolutional Neural Network based Live Object Recognition 
System as Blind Aid uses a different dataset for normal objects and not vehicles.
  Benjdira et. al (2019) on Car Detection using Unmanned Aerial Vehicles: Comparison 
between Faster R-CNN and YOLOv3 attempts to look at vehicles from a sky view unlike mine.

Methodology

My methodology is to create strong CNN models that can accurately predict cars from their 
images. Strong CNN models need a large dataset, strong architecture, and a long enough
time to train. I began the project by looking for a dataset and settled on the Stanford dataset;
a collection of 197 different classes of car separated by car make, model, and year using over 16000 images.
The Stanford dataset contains “16,185 images of 197 classes of cars. The data is split into 8,144 
training images and 8,041 testing images.” (Krause et al. 2013) The large dataset contains images 
with varying size but CNNs require datasets to be uniform. Resizing the images into sizes of 224
by 224 distorts the data but creates a uniform model. In my initial assessment I used a basic CNN
model shortly trained on only 5 epochs. Increasing the training to 50 epochs increased the accuracy 
but not to an acceptable level. The model summary can be seen below in Figure 1.
![image](https://user-images.githubusercontent.com/71231702/213226308-72023e15-b5be-4986-ad2b-cb5b8472249a.png)
Figure 1. Basic CNN model summary.
The product and need for higher accuracy from the models can be seen in the demonstration. To 
demonstrate the models, I created a prediction program that would load in the models and use them 
to predict the label of an image. There we can see a clear success or failure of the model. To 
achieve success with my models I needed to move to a different framework.
ResNet50, a deep neural network framework is a popular and accurate framework that can 
accurately be used for car image detection (Tahir et. al, 2021). ResNet50 is a variant of the ResNet
model with 48 convolutional layers that allows for ultra-deep neural networks. Creating a new 
model with a stronger architecture like ResNet50, with adequate time to train, produced a highly
accurate model. I then used this model for my prediction model.

Results

![image](https://user-images.githubusercontent.com/71231702/213226951-66f15f51-00e0-49d8-955b-d55bed24b84d.png)
Figure 2.  Basic CNN (5 epochs) Training Accuracy and Loss.

In my initial test the trendline seen in Figure 2 indicates a clear growth rate for the basic CNN model. The model is clearly training on the Stanford dataset, getting more accurate, and decreasing its loss. I then increase the epochs to 50, which is my standard for my other model. The basic CNN model trained on 50 epochs demonstrates a significant accuracy growth and decrease in loss. However, at only ~35% accuracy with 50 epochs It would take another 50 epochs to achieve ~70% accuracy, which is almost usable for the predicter if it follows the trendline represented below in Figure 3. 

![image](https://user-images.githubusercontent.com/71231702/213227193-a8544d62-0287-4bbe-b9d7-76264edc1fe1.png)
Figure 3.  Basic CNN (50 epochs) Training Accuracy and Loss.
Using the prediction program, on the basic CNN model at 50 epochs, I demonstrate the failure of the model to accurately predict the correct vehicle, as seen in Figure 4. The prediction program uses the basic CNN model to predict the image belongs to class 32 with 9.48% confidence when the image belongs to class 116. This specific image can be seen in Figure 5.
![image](https://user-images.githubusercontent.com/71231702/213227445-26f71755-718f-4bc6-b073-8adf7fb010dd.png)
Figure 4.  Basic CNN (50 epochs) Prediction.
![image](https://user-images.githubusercontent.com/71231702/213227513-5bba588f-1d71-4d82-a2f5-e544610509d4.png)
Figure 5. Prediction Program Result Image.
A ~35% accuracy model consistently fails to produce accurate results unless we use a better CNN model and train with a larger epoch set. Moving to the ResNet50 model we find a highly accurate model (~90%) after only ~20 epochs. The superior framework quickly builds accuracy and decreases loss as seen in Figure 6 below. 
![image](https://user-images.githubusercontent.com/71231702/213227686-22eb92c3-e142-4db2-a74b-e2717db870d2.png)
Figure 6.  ResNet50 CNN (50 epochs) Training Accuracy and Loss.
However, using this model with my prediction program I receive an unexpected result as seen in Figure 7. The model produces an incorrect result with a lower percent confidence. 
![image](https://user-images.githubusercontent.com/71231702/213227832-ca59356e-cbd2-4e89-a49a-695b521ab5a6.png)
Figure 7.  Basic CNN (50 epochs) Prediction.

Conclusion

I present an image classification convolution neural network and prediction program for determining the year, make, and model of a car using the Stanford dataset. I begin with simple CNN models like my Basic CNN model and move on to a more complex framework. Upgrading the framework of my CNN models I achieve better accuracy and lower loss. I attempt to use these models along with the prediction program to accurately classify images but fail even when using the ResNet50 framework. The ResNet50 model produces high accuracies but fails to classify images. The disassociation between my model accuracy and my model prediction is not yet understood. Moving forward I will attempt to solve this mystery and expand into object detection with YOLO or Faster-CNN.





Plan for Evaluation / Demonstration

My plan for evaluation includes measuring the accuracy and loss metrics from training, validation, and testing. Using this data, I will compare my CNN models. Achieving an accuracy of over 80% would be ideal. My first plan for demonstration includes running an image of a car through the model and judge the label given to the image by the model. My second plan for demonstration includes judging the live vehicle image identification algorithm if it outputs the correct label.
