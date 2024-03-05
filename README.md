# Multi-Class Dog Breed Image Classification using TensorFlow

This project uses machine learning, deep learning, and transfer learning to build a multi-class image classifier using TensorFlow 2.0 and TensorFlow Hub.

## Problem

Identify the breed of a dog given an image of the dog.

## Data

The data comes from Kaggle's dog breed identification competition: https://www.kaggle.com/competitions/dog-breed-identification/data

It is a collection of images of dogs broken into training and testing sets of over 10,000 images each. There are 120 different breeds of dogs (classes) in the training data.

## Evaluation

The evaluation is a file with prediction probabilities for each dog breed of each test image as outlined here: https://www.kaggle.com/competitions/dog-breed-identification/overview/evaluation

## Data Preparation

The first step was to convert all of the images into Tensors to be compatible with machine learning tools.

I also looked at the distribution of dog breeds in the training set, visualized in Figure 1.

> ![image](https://github.com/nwferreri/dog-breed-image-classification/assets/112211174/8defd95d-f8c7-47da-94a3-8e83fcf2a9f9)
> 
> **Figure 1**: The number of images present for each of the 120 dog breeds in the training data. The median number of images was 82. The data seems to be fairly well balanced across the different breeds.

A validation data set was created from a subset of the training data. I also started off only working with 1000 images to speed up testing.

A function was created to turn images into tensors with 3 normalized color channels and an image size of (224, 224).

Next, I created a function to process all data in batches. This function can be used for training, validation, and test data sets.

Finally, batches were visualized, an example of which can be seen in Figure 2.

> ![image](https://github.com/nwferreri/dog-breed-image-classification/assets/112211174/f277d91d-ea5e-420b-8d98-2426bfce117b)
> 
> **Figure 2**: A visualization of 25 images in a batch of training data.

## Initial Model Training and Evaluation

I selected the MobileNet V2 model found here: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5. With it, I built a Keras deep learning model.

I created 2 callback functions: one using TensorBoard to create training logs for the model and an early stopping callback to prevent the model from overfitting.

An initial model was trained on the training subset of images. I then used that model to make predictions on the validation data.

To assist with evaluating those predictions, I created functions to help visualize the image and the confidence intervals for the top 10 breed predictions. I added color to indicate if the model correctly classified the dog in the image. See Figure 3 for a few examples.

> ![image](https://github.com/nwferreri/dog-breed-image-classification/assets/112211174/ceb08e6b-4c43-4957-8020-f198d82b0bdd)
> 
> **Figure 3**: Several example visualizations of the initial model's predictions on the validation data. The image is shown on the left, with the predicted label, the confidence interval, and the correct label shown above. The color indicates if the model was correct (green) or not (red). On the right, the top 10 breed confidence intervals shown with the correct label being colored green.

This initial model was evaluated and found to have an accuracy of 67.5% on the validation data.

## Full Model Training

Next, a model was trained on the full training data set of 10,000+ images. The model was saved for future use.

Then the model was used to make predictions on the test data, which were also formatted for Kaggle submission an saved.

## Predicting on Custom Images

Last, I created a pipeline that allows using the full trained model to make predictions on new custom images from the user and visualize the results, an example of which can be seen in Figure 4.

> ![image](https://github.com/nwferreri/dog-breed-image-classification/assets/112211174/d6cf67c0-12dc-4ecd-9d92-2d6c9342f027)
> 
> **Figure 4**: An example of predictions made on custom images using the full trained model.
