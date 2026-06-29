import os
import sys
import argparse

# Add workspace directory to python path for easy imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tensorflow as tf
from backend.app.ml.initialize_models import build_age_model
from backend.ml_training.preprocess import get_data_pipelines
from backend.app.core.logging import logger

def train_age_classifier(dataset_dir: str, epochs: int, batch_size: int, output_path: str):
    logger.info(f"Starting Age Group Classifier training on dataset from: {dataset_dir}")
    
    # Get datasets
    try:
        train_ds, val_ds = get_data_pipelines(dataset_dir, batch_size=batch_size, task="age")
    except Exception as e:
        logger.error(f"Failed to prepare data pipeline: {str(e)}")
        logger.info("Please make sure the dataset path is correct and contains valid UTKFace format files.")
        return
        
    # Build age group model architecture
    model = build_age_model()
    model.summary()
    
    # Callback to save best model weights
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        output_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    logger.info(f"Training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint]
    )
    
    logger.info(f"Age Group classification model training completed. Best model saved to: {output_path}")
    
    # Evaluate model
    val_loss, val_acc = model.evaluate(val_ds)
    logger.info(f"Final Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Age Group Classification Model")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="./dataset_mock", 
        help="Path to UTKFace dataset directory"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument(
        "--output", 
        type=str, 
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "ml", "models", "age_model.h5")),
        help="Output filepath for trained model (.h5)"
    )
    
    args = parser.parse_args()
    
    # Ensure parent output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    train_age_classifier(args.dataset, args.epochs, args.batch_size, args.output)
