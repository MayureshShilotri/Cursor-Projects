# MNIST Character Recognition

A Python project for recognizing handwritten digits using the MNIST dataset and deep learning.

## Project Overview
This project implements a neural network model to recognize handwritten digits (0-9) using the MNIST dataset. The model is built using TensorFlow and includes data preprocessing, model training, and evaluation components.

## Features
- Data loading and preprocessing
- Neural network model implementation
- Model training and evaluation
- Visualization of results
- Prediction on new images

## Requirements
- Python 3.8+
- TensorFlow 2.13.0
- NumPy 1.24.3
- Matplotlib 3.7.2
- scikit-learn 1.3.0
- pandas 2.0.3

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model:
   ```bash
   python train_model.py
   ```

## Project Structure
- `train_model.py`: Main script for training the model
- `model.py`: Neural network model architecture
- `data_loader.py`: Data loading and preprocessing
- `utils.py`: Utility functions
- `requirements.txt`: Project dependencies

## Model Architecture
The model uses a Convolutional Neural Network (CNN) architecture with:
- Convolutional layers
- Max pooling layers
- Dense layers
- Dropout for regularization

## Performance
The model achieves approximately 99% accuracy on the test set.

## License
MIT License 