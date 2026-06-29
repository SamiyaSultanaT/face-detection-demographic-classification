import cv2
import numpy as np
import tensorflow as tf
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.ml.initialize_models import initialize_ml_models

# List of age categories in order of classes (index 0 to 7)
AGE_GROUPS = ["0-2", "4-6", "8-12", "15-20", "21-30", "31-40", "41-50", "51+"]

class AgeGenderPredictor:
    def __init__(self):
        # Initialize baseline models if they are not found
        initialize_ml_models()
        
        try:
            logger.info("Loading Gender Classification Model...")
            self.gender_model = tf.keras.models.load_model(settings.GENDER_MODEL_PATH)
            logger.info("Loading Age Group Classification Model...")
            self.age_model = tf.keras.models.load_model(settings.AGE_MODEL_PATH)
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load ML models: {str(e)}")
            raise

    def preprocess_image(self, face_np: np.ndarray) -> np.ndarray:
        """
        Preprocesses a cropped BGR face image for the CNN model input.
        - Resizes to 96x96
        - Converts BGR to RGB
        - Normalizes pixel values [0, 1]
        - Adds batch dimension
        """
        # Resize to model input size
        face_resized = cv2.resize(face_np, (96, 96))
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        # Normalize
        face_normalized = face_rgb.astype(np.float32) / 255.0
        # Expand dimensions to (1, 96, 96, 3)
        return np.expand_dims(face_normalized, axis=0)

    def predict(self, face_np: np.ndarray):
        """
        Predicts gender and age group from a cropped face.
        
        Returns:
            gender: str ("Male" or "Female")
            gender_confidence: float
            age_group: str (e.g. "21-30")
            age_confidence: float
        """
        try:
            input_tensor = self.preprocess_image(face_np)
            
            # Predict gender
            # Output is a single probability value from sigmoid
            gender_pred = self.gender_model.predict(input_tensor, verbose=0)[0][0]
            if gender_pred >= 0.5:
                gender = "Female"
                gender_confidence = float(gender_pred)
            else:
                gender = "Male"
                gender_confidence = float(1.0 - gender_pred)
                
            # Predict age group
            # Output is a softmax vector of length 8
            age_pred = self.age_model.predict(input_tensor, verbose=0)[0]
            age_class_idx = int(np.argmax(age_pred))
            age_group = AGE_GROUPS[age_class_idx]
            age_confidence = float(age_pred[age_class_idx])
            
            return gender, gender_confidence, age_group, age_confidence
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            # Default fallback predictions on error
            return "Male", 0.5, "21-30", 0.125

# Instantiate global predictor instance
predictor = AgeGenderPredictor()
