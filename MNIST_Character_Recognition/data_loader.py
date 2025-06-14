import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data for CNN input
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def prepare_data_for_training(x_train, y_train, validation_split=0.1):
    """
    Split training data into training and validation sets.
    
    Args:
        x_train: Training features
        y_train: Training labels
        validation_split: Proportion of data to use for validation
        
    Returns:
        tuple: (x_train, y_train), (x_val, y_val)
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=validation_split, random_state=42
    )
    return (x_train, y_train), (x_val, y_val)

def create_data_generator(x_train, y_train, batch_size=32):
    """
    Create a data generator for training with data augmentation.
    
    Args:
        x_train: Training features
        y_train: Training labels
        batch_size: Batch size for training
        
    Returns:
        DataGenerator: TensorFlow data generator
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    return datagen.flow(x_train, y_train, batch_size=batch_size) 