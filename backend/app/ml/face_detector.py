import cv2
import numpy as np
from backend.app.core.logging import logger

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error(f"Error loading Haar Cascade from: {cascade_path}")
            raise RuntimeError("Could not load OpenCV Haar Cascade XML.")
        logger.info("OpenCV Haar Cascade loaded successfully.")

    def detect_faces(self, image_np: np.ndarray):
        """
        Detects faces in a BGR image.
        
        Args:
            image_np: A numpy array representing the image (BGR format).
            
        Returns:
            A list of dictionaries containing the bounding box coordinates [x, y, w, h] 
            and the cropped face sub-image.
        """
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            # Detect multi-scale faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )
            
            results = []
            height, width, _ = image_np.shape
            
            for (x, y, w, h) in faces:
                # Add a padding of 15% to capture the whole head (including hair/chin)
                pad_w = int(w * 0.15)
                pad_h = int(h * 0.15)
                
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(width, x + w + pad_w)
                y2 = min(height, y + h + pad_h)
                
                cropped_face = image_np[y1:y2, x1:x2]
                
                results.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "cropped_face": cropped_face
                })
                
            return results
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            # Fall back to empty list on failure
            return []

face_detector = FaceDetector()
