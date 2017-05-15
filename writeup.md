#**Behavioral Cloning** 

##Writeup Boris Kamenjasevic

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted and Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model structure is as follows:

First every picture is downscaled by 2 to improve training time.

 - Cropping layer to remove top 25 and bottom 10 pixels
 - Lambda layer to normalise the data and shift by the mean
 - Convolution layer, depth=12, kernel=3,3
 - Max Pooling, 2x2
 - Convolution layer, depth=24, kernel=3,3
 - Max Pooling, 2x2
 - Convolution layer, depth=36, kernel=3,3
 - Convolution layer, depth=48, kernel=3,3
 - Flattening
 - Fully Connected 300
 - Relu Activation
 - Dropout, rate = 0.25
 - Fully Connected 20
 - Relu Activation
 - Fully Connected 1

####2. Attempts to reduce overfitting in the model

The model contains max pooling layers each after the first two convolutional layers to reduce overfitting und overinterpreting image features.
Also I implement dropouts. After the two first fully connected layer there are dropouts with rates of 0.25 and 0.1.

The data set was divided in training (80%) and validation data (20%) (model.py line 26). Testing was done using the simulator.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 103).

####4. Appropriate training data

For training data I used 7 laps of center line driving on the first track and one on the more advanced track which were collected using a gamepad to get smoother steering wheel angles.
In the training set left and right images are included and a steering angle offset of +/-0.2 is added as matching values.
Also flipped versions of the center images are included to generalise the model better.
Additionally recovery runs were recorded since especially the bridge seemed to be a difficult task for the model.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The model architecture was designed to stay close to exisiting models that are already proven and simplify from there to fit the task.

The Nvidia network was used an example in this case. Since it is already proven to work, I thought it is a good starting point.
From there I reduced the depth from 64 to 48 and removed one convolutional layer.

Since there was an low mean squared error on the training data set and high MSE on the validation it was clear that the model was overfitting.
After adding max pooling layers and dropouts the overfitting was reduced.

####2. Final Model Architecture

See section 1.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 5 laps on track one using center lane driving.
First I tried fitting models with only 2-3 laps which led nowhere.

Then I recorded recovery maneuvers so the model knows how to react when it approaches the limits of the track.

To better generalise the model I flipped the center images.

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was5Z as evidenced by trying out. More epoch led to increasing MSE on the validation set.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
