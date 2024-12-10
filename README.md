# CYSE 420 Project: Cybersecurity Machine Learning Models

## Overview
This project focuses on applying machine learning models to the **KDD Cup 1999 dataset**, which is used for network intrusion detection. The task is to classify network traffic into "normal" or "attack" categories. The project implements five distinct machine learning models and compares their performance using several evaluation metrics.

### Key Objectives:
- Implement and evaluate at least five different machine learning models on the KDD dataset.
- Explore ensemble learning techniques to improve predictive accuracy.
- Compare the performance of individual models with ensemble models.

## Dataset
The dataset used in this project is the **KDD Cup 1999 dataset**. It contains network traffic data with features like:
- **Duration**: The duration of the connection.
- **Protocol Type**: The type of protocol (e.g., TCP, UDP).
- **Service**: The network service (e.g., HTTP, FTP).
- **Flag**: The connection status flag.
- **Source/Destination Bytes**: The number of bytes sent or received during the connection.
- **Label**: The class label indicating whether the connection is "normal" or an attack (e.g., "doS", "probe", "R2L", "U2R").

The dataset is divided into:
- **Training data**: `kddtrain+.data`
- **Test data**: `kddtest+.data`

## Models Implemented
The following five models are implemented and evaluated:
1. **Naive Bayes**: A probabilistic classifier based on Bayes' Theorem.
2. **Support Vector Machine (SVM)**: A classifier that separates data points using an optimal hyperplane.
3. **Random Forest**: An ensemble method using multiple decision trees.
4. **Decision Tree**: A tree-based model for classification.
5. **Neural Network**: A deep learning model using multi-layer perceptrons.

### Ensemble Learning
In addition to individual models, we explore ensemble techniques (like bagging and boosting) to combine predictions and improve model accuracy.

## Evaluation Metrics
The models are evaluated based on the following metrics:
- **Accuracy**: The proportion of correct predictions.
- **F1-Score**: The harmonic mean of precision and recall, suitable for imbalanced datasets.
- **Precision**: The percentage of true positive predictions among all positive predictions.
- **Recall**: The percentage of true positive predictions among all actual positives.
- **AUC-ROC**: The Area Under the Receiver Operating Characteristic Curve, measuring the trade-off between true positive and false positive rates.
