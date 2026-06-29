import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from backend.app.core.config import settings
from backend.app.core.logging import logger

def build_gender_model():
    """
    Builds the Gender Classification CNN architecture.
    Inputs: 96x96x3 images.
    Output: Binary sigmoid (1 neuron: Close to 0 -> Female, Close to 1 -> Male).
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_age_model():
    """
    Builds the Age Group Classification CNN architecture.
    Inputs: 96x96x3 images.
    Output: Softmax over 8 age categories.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def initialize_ml_models(force: bool = False):
    """
    Ensures both gender and age group model weight files exist.
    If they do not exist, creates models with initialized weights so the API is immediately runnable.
    """
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    
    # 1. Gender Model
    if force or not os.path.exists(settings.GENDER_MODEL_PATH):
        logger.info("Initializing baseline Gender Classification Model...")
        gender_model = build_gender_model()
        gender_model.save(settings.GENDER_MODEL_PATH)
        logger.info(f"Saved baseline Gender Model to {settings.GENDER_MODEL_PATH}")
    else:
        logger.info("Gender Model file already exists.")

    # 2. Age Model
    if force or not os.path.exists(settings.AGE_MODEL_PATH):
        logger.info("Initializing baseline Age Group Classification Model...")
        age_model = build_age_model()
        age_model.save(settings.AGE_MODEL_PATH)
        logger.info(f"Saved baseline Age Model to {settings.AGE_MODEL_PATH}")
    else:
        logger.info("Age Model file already exists.")

if __name__ == "__main__":
    initialize_ml_models()
