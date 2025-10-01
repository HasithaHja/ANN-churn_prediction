# Customer Churn Prediction

A machine learning project that predicts customer churn using an Artificial Neural Network (ANN) built with TensorFlow/Keras. The project includes a Streamlit web application for interactive predictions.

## Project Overview

This project implements a binary classification model to predict whether a bank customer will churn (leave the service) based on various features such as credit score, geography, gender, age, balance, and other account-related information.

## Features

- **Deep Learning Model**: 3-layer neural network with ReLU and sigmoid activations
- **Interactive Web App**: Streamlit-based UI for real-time predictions
- **Data Preprocessing**: Includes label encoding, one-hot encoding, and feature scaling
- **Model Monitoring**: TensorBoard integration for training visualization
- **Early Stopping**: Prevents overfitting during training

## Project Structure

```
.
├── app.py                          # Streamlit web application
├── experiments.ipynb               # Jupyter notebook with full ML pipeline
├── model.h5                        # Trained neural network model
├── label_encoder_gender.pkl        # Gender label encoder
├── onehot_encoder_geo.pkl          # Geography one-hot encoder
├── scaler.pkl                      # Standard scaler for feature normalization
└── logs/                           # TensorBoard logs directory
```

## Model Architecture

- **Input Layer**: 12 features (after preprocessing)
- **Hidden Layer 1**: 64 neurons, ReLU activation
- **Hidden Layer 2**: 32 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation (binary classification)
- **Total Parameters**: 2,945

## Installation

1. Install required dependencies:
```bash
pip install streamlit numpy tensorflow scikit-learn pandas
```

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

The app will launch in your browser where you can input customer information:
- Geography (France, Germany, Spain)
- Gender (Male, Female)
- Age (18-92)
- Balance
- Credit Score
- Estimated Salary
- Tenure (0-10 years)
- Number of Products (1-4)
- Has Credit Card (Yes/No)
- Is Active Member (Yes/No)

The model will output:
- Churn probability (0-1)
- Prediction: "likely to churn" or "not likely to churn" (threshold: 0.5)

### Training the Model

The complete training pipeline is available in `experiments.ipynb`. The notebook includes:
1. Data loading from Kaggle dataset
2. Data preprocessing and feature engineering
3. Model training with callbacks
4. Model evaluation and saving

## Dataset

The project uses the "Churn Modelling" dataset from Kaggle (`shrutimechlearn/churn-modelling`), which contains 10,000 customer records with the following features:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Exited (target variable)

## Model Performance

- **Training Accuracy**: ~87%
- **Validation Accuracy**: ~85-86%
- **Training stopped at**: Epoch 12 (early stopping due to validation loss plateau)

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **Scikit-learn**: Preprocessing and encoding
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **TensorBoard**: Training visualization

## License

This project is for educational purposes.
