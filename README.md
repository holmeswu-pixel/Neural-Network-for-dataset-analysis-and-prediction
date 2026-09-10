# Neural-Network-for-dataset-analysis-and-prediction
This repository contains three distinct neural network models designed to analyze individual datasets and solve different types of predictive tasks. All networks are implemented from scratch in **Python** using the **PyTorch** framework.

## Key Features
* **Multi-Task Implementations**: Covers regression/classification tasks and image recognition.
* **PyTorch Framework**: Leverages modern deep learning workflows (DataLoaders, loss functions, and optimizers).
* **Concepts and Applications**: Utilizes the basic concepts of deep learning (loss function, activation function, learning rate, normalization, etc.).

## 🧠 Model Overview

### 1. Score Prediction Network
* **Type**: Simple Feedforward Neural Network (MLP)
* **Objective**: Predict a continuous continuous score based on selected features from the dataset.
* **Core Components**: Fully connected layers (`nn.Linear`), ReLU activation functions, and MSE Loss.

### 2. Risk Prediction Network
* **Type**: Simple Feedforward Neural Network (MLP)
* **Objective**: Predict risk levels or probabilities from selected input parameters.
* **Core Components**: Fully connected layers with a Sigmoid or Softmax output layer, optimized using Cross-Entropy or Binary Cross-Entropy Loss.

### 3. MNIST Image Recognition Network
* **Type**: Convolutional Neural Network (CNN)
* **Objective**: Classify handwritten digits (0–9) using the built-in MNIST dataset from `torchvision`.
* **Core Components**: Convolutional layers (`nn.Conv2d`), Max Pooling (`nn.MaxPool2d`), and fully connected layers for final classification.

## 🛠️ Prerequisites & Installation

To run these models locally, ensure you have Python installed. Make sure the necessary packages are installed to conduct the program smoothly (torch, torchvision).
