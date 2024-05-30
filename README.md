# Dog-vs-Cat-Image-Classification
This repository contains a deep learning project for classifying dog and cat images using a pre-trained MobileNetV2 model. The project includes data preprocessing, image resizing, model training, and a predictive system to identify whether an image is a dog or a cat.

![image](https://github.com/lekshmiij/Dog-vs-Cat-Image-Classification/assets/141242851/7b4e6fd3-359f-4ebc-8060-0459da0d7be7)

This project involves creating a machine learning model to classify images of dogs and cats. The process can be broken down into several key steps:


### Libraries and Initial Image Display:
Necessary libraries are imported, including NumPy, PIL, Matplotlib, and Scikit-Learn.
Example images of a dog and a cat are displayed using Matplotlib.

### Data Exploration and Labeling:
The filenames in the dataset are inspected to understand the structure.
The number of dog and cat images is counted based on filename prefixes.

### Image Resizing:
Images are resized to 224x224 pixels and converted to RGB format.
Resized images are saved to a new directory.

### Label Creation:
Labels are created for the images: 'Cat' as 0 and 'Dog' as 1.
The first few labels and the count of each class are printed.

### Image Conversion to Numpy Arrays:
Resized images are converted into numpy arrays using OpenCV.
The resulting array is inspected for shape and type.

### Train-Test Split:
The dataset is split into training and testing sets using an 80-20 split.
The shapes of the training and testing datasets are printed.

### Data Scaling:
Pixel values are scaled to the range [0, 1] by dividing by 255.

### Building the Neural Network:
TensorFlow and TensorFlow Hub are used to build a neural network with a pre-trained MobileNetV2 model.
The model includes a dense layer for classification and is compiled with the Adam optimizer and sparse categorical cross-entropy loss.

### Training and Evaluation:
The model is trained on the scaled training dataset for 5 epochs.
The model's performance is evaluated on the test dataset, and the test loss and accuracy are printed.

### Predictive System:
A predictive system is implemented to classify new images as either a dog or a cat.
The input image is resized, scaled, and reshaped before being passed to the model for prediction.
The predicted label is printed, indicating whether the image represents a cat or a dog.

_This process involves downloading and preparing the dataset, exploring and visualizing the data, preprocessing images, building and training a neural network model, and implementing a system for real-time image classification. The use of a pre-trained model (MobileNetV2) helps leverage transfer learning for effective image classification with a relatively small dataset._
