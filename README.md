# Logistic-Regression-from-Scratch
This is a Logistic Regression model that I coded from scratch. Actively trying to optimize it and add features. **End goal: be able to fine-tune hyperparameters and insert datasets.**

This project implements **Logistic Regression** from scratch using Python, without relying on machine learning libraries like `scikit-learn`. The goal is to understand the underlying principles and math behind logistic regression, as well as gain hands-on experience in implementing it.

## Project Overview

Logistic Regression is a statistical method used for binary classification tasks. It predicts the probability of an instance belonging to a particular class, making it a valuable algorithm for problems like spam detection, medical diagnoses, and more. This project walks through the process of implementing the algorithm step by step and training it on real-world datasets.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Steps Involved](#steps-involved)
- [How to Run the Project](#how-to-run-the-project)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- Logistic Regression Model implemented from scratch using Python.
- Demonstrates the core concepts of **sigmoid function**, **cost function**, and **gradient descent**.
- Includes an example dataset for training and evaluation.
- Step-by-step explanation of the model's building blocks.

## Technologies

- **Python 3.x**: The primary programming language for the implementation.
- **NumPy**: Used for matrix operations and handling large datasets efficiently.
- **Matplotlib** (optional): For visualizing the results (if implemented).

## Dataset

The dataset used for training the Logistic Regression model is [describe your dataset here, e.g., "the famous Iris dataset" or "a custom dataset for binary classification"].

Ensure that the dataset is preprocessed and divided into training and test sets for evaluation.

## Implementation Details

This project consists of the following key components:

1. **Sigmoid Function**: Computes the probability output for a given input.
2. **Cost Function**: Measures the performance of the model and guides optimization.
3. **Gradient Descent**: Optimizes the model parameters by iteratively minimizing the cost function.
4. **Training the Model**: Using gradient descent to optimize the weights and bias.
5. **Model Evaluation**: Evaluating the model's performance using accuracy, precision, recall, etc.

## Steps Involved

1. **Data Preprocessing**: Clean and prepare the dataset for training.
2. **Feature Engineering**: Normalize or scale the features (if necessary).
3. **Model Training**: Train the logistic regression model using gradient descent.
4. **Model Evaluation**: Evaluate the model using appropriate performance metrics like accuracy and confusion matrix.
5. **Prediction**: Make predictions on new, unseen data.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/larissapaola1/Logistic-Regression-from-Scratch.git
   cd Logistic-Regression-from-Scratch

2. Install required dependencies:
   
  pip install -r requirements.txt

3. Run the Python script to train the logistic regression model:
  python logistic_regression.py

# Acknowledgments
Andrew Ng's Machine Learning Course for the inspiration behind the implementation. https://www.coursera.org/learn/machine-learning
