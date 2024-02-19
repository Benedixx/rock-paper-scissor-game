# Rock paper scissor object detection

## Project Overview

This project aims to make AI as referee for rock paper scissor game using webcam as input, i use yoloV5 algorithm to detect user hand and determine the hand of left side and right side. the result is shown when the left side and right side hand is captured.

## Dataset

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data). It consists of brain MRI images with labeled tumor types. Ensure that you have downloaded and preprocessed the dataset as required before running the code or you can use the dataset that i provided on this repository.

Dataset contains :<br>
number of glioma training data: 702 <br>
number of meningioma training data: 704<br>
number of training data without tumors: 280<br>
number of pituitary training data: 576<br>

number of glioma validation data: 199<br>
number of meningioma validation data: 209<br>
number of without tumors validation data: 158<br>
number of pituitary validation data: 268<br>


## Model Architecture

The deep learning model is built using the TensorFlow Keras libraries. The architecture typically involves:

- Preprocessing the MRI images (resizing, normalization, augmentation, etc.).
- Creating a Convolutional Neural Network (CNN) with several convolutional and pooling layers and adding dense layers with 4 output.
- Compiling the model with an appropriate loss function and optimizer.
- Training the model on the dataset.

You can find the detailed architecture and code in the Jupyter Notebook or Python script provided in this repository.

## Training Result
### training and validation acccuracy graph
![image](https://github.com/Benedixx/Brain-Tumor-Classification-tensorflow/assets/97221880/8d4be853-37a6-4ded-af6f-de1557620553)

### training and validation loss graph
![image](https://github.com/Benedixx/Brain-Tumor-Classification-tensorflow/assets/97221880/85623827-4d56-41fd-9b58-9eb88f9dbdae)

### training and validation acccuracy on last epoch
```bash
Epoch 25/25
194/194 [==============================] - 31s 158ms/step - loss: 0.0020 - accuracy: 0.9990 - val_loss: 1.3206e-04 - val_accuracy: 1.0000
```
