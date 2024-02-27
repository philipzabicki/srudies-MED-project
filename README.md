# Income and Voice Prediction Models

## Overview

This project solves a final assignment for a subject in my studie. It leverages various machine learning algorithms to predict income levels and analyze voice data. It includes models built with techniques such as bagging, random forests, boosting (AdaBoost), and neural networks, offering a comprehensive approach to predictive analytics.

## Datasets

- **`income.csv`**: Used for building income prediction models.
- **`voice.csv`**: Utilized for a neural network model to classify voice data.

## Features

- **Data Preprocessing**: Implements preprocessing steps including handling missing values and splitting data into training and test sets.
- **Model Evaluation**: Functions for calculating accuracy, plotting ROC curves, and evaluating model performance are included.
- **Model Optimization**: Explores hyperparameter tuning for bagging, random forests, and neural networks to optimize performance.
- **Visualization**: Features visualizations such as ROC curves, variable importance plots, and neural network diagrams for intuitive analysis.

## Libraries Used

The project uses R programming libraries such as `ROCR`, `ipred`, `caTools`, `rattle`, `rpart`, `caret`, `randomForest`, `adabag`, `gbm`, `pROC`, `neuralnet`, `nnet`, and `NeuralNetTools`.

## Getting Started

1. **Clone the Repository**: Clone this repo to get started with the project.
2. **Install Required Libraries**: Make sure all required R libraries are installed.
3. **Run the Scripts**: Execute the scripts to train the models on the provided datasets. Adjust parameters and datasets as needed.

## Model Summaries

- **Bagging and Random Forests**: Showcases ensemble methods for improving prediction accuracy.
- **Boosting (AdaBoost)**: Employs AdaBoost for enhanced model performance.
- **Neural Networks**: Demonstrates the application of neural networks on voice data classification.
