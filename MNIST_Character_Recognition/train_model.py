import tensorflow as tf
from data_loader import load_mnist_data, prepare_data_for_training, create_data_generator
from model import create_model, create_callbacks
from utils import plot_training_history, evaluate_model

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Load and preprocess data
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Prepare data for training
    print("Preparing data for training...")
    (x_train, y_train), (x_val, y_val) = prepare_data_for_training(x_train, y_train)
    
    # Create data generator
    train_generator = create_data_generator(x_train, y_train)
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        steps_per_epoch=len(x_train) // 32
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, x_test, y_test)
    
    # Save the final model
    print("\nSaving model...")
    model.save('mnist_model.h5')
    print("Model saved as 'mnist_model.h5'")

if __name__ == "__main__":
    main() 