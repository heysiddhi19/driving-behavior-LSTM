# driving-behavior-LSTM
# Driver Behavior Analysis using LSTM

A deep learning project that analyzes smartphone sensor telemetry (accelerometer and gyroscope) to classify driving behavior. The model evaluates temporal driving patterns and generates a safety score to help reduce traffic accidents.

## Overview
This project uses a Long Short-Term Memory (LSTM) neural network to process sequential sensor data (X, Y, Z axes for both acceleration and angular velocity). By analyzing sliding windows of approximately 2 seconds (20 time-steps), the system classifies the driver's actions and outputs a real-time evaluation rating from 1 to 5 Stars

Based on the score, drivers are categorized into four profiles: Excellent, Good, Average, and Risky.

## Tech Stack
- **Language:** Python 3.12
- **Deep Learning:** TensorFlow / Keras (Sequential LSTM)
- **Data Processing:** Pandas, NumPy, Scikit-Learn
- **Visualization:** Matplotlib, Seaborn

## Repository Structure
- `Mini_project_2 (1).ipynb`: The main notebook containing the data preprocessing pipeline, sliding window sequence generation, model architecture, and training loop.
- `driving_lstm_model.keras`: The trained Keras model file.
- `data_scaler.pkl`: The saved StandardScaler object. Essential for normalizing raw sensor inputs before running inference.
- `Mini Project 2 Report.pdf`: Detailed project documentation, including methodology, challenges faced (e.g., handling high-frequency sensor noise), and future scope.

## Model Performance
The model was trained over 10 epochs (batch size 64) using a T4 GPU.
- **Validation Accuracy:** 90.77%
- **Final Loss:** 0.2125

## How to Use
To run inferences on new sensor data, ensure you load both the model and the scaler. The raw data must be scaled before passing it to the LSTM model.

```python
import pickle
from tensorflow.keras.models import load_model

# 1. Load the scaler
with open('data_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2. Load the trained model
model = load_model('driving_lstm_model.keras')

# Normalize your raw data and predict...
