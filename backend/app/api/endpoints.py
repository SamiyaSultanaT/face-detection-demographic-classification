import os
import uuid
import base64
import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.logging import logger
from backend.app.ml.face_detector import face_detector
from backend.app.ml.predictor import predictor
from backend.app.services import db_service, dashboard_service
from backend.app.schemas.history import PredictionHistoryResponse
from backend.app.schemas.dashboard import DashboardStats

router = APIRouter()

# Schema for webcam payload
class WebcamPayload(BaseModel):
    image: str  # Base64 string (data:image/jpeg;base64,...)

# Schema for prediction detection item
class DetectionResult(BaseModel):
    box: List[int]
    gender: str
    gender_confidence: float
    age_group: str
    age_confidence: float
    confidence: float

# Schema for final response
class PredictionResponse(BaseModel):
    image_name: str
    detections: List[DetectionResult]

def process_and_predict_image(image_np: np.ndarray, filename_prefix: str, db: Session) -> PredictionResponse:
    """
    Core function to run face detection and predictions.
    Saves the image to the uploads directory and logs predictions in the database.
    """
    if image_np is None or image_np.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid image data."
        )

    # Generate a unique filename and save the uploaded image
    unique_filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(settings.UPLOAD_DIR, unique_filename)
    cv2.imwrite(filepath, image_np)

    # Detect faces
    faces = face_detector.detect_faces(image_np)
    detections = []

    for idx, face_data in enumerate(faces):
        box = face_data["box"]
        cropped_face = face_data["cropped_face"]
        
        # Run inference
        gender, gender_conf, age_group, age_conf = predictor.predict(cropped_face)
        # Average confidence score
        overall_conf = (gender_conf + age_conf) / 2.0
        
        # Save to database
        try:
            db_service.save_prediction(
                db=db,
                image_name=unique_filename,
                predicted_gender=gender,
                predicted_age_group=age_group,
                confidence=overall_conf
            )
        except Exception as e:
            logger.error(f"Error saving detection to DB: {str(e)}")

        detections.append(
            DetectionResult(
                box=box,
                gender=gender,
                gender_confidence=round(gender_conf, 4),
                age_group=age_group,
                age_confidence=round(age_conf, 4),
                confidence=round(overall_conf, 4)
            )
        )

    return PredictionResponse(image_name=unique_filename, detections=detections)


@router.post("/predict-image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload an image, detect faces, predict age/gender, and save to history.
    """
    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload JPG, PNG or WEBP."
        )
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return process_and_predict_image(img, "upload", db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /predict-image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred processing the image: {str(e)}"
        )


@router.post("/predict-webcam", response_model=PredictionResponse)
async def predict_webcam(payload: WebcamPayload, db: Session = Depends(get_db)):
    """
    Accepts base64 webcam frame, runs predictions, and saves to history.
    """
    try:
        # Base64 string check and parsing
        base64_str = payload.image
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
            
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return process_and_predict_image(img, "webcam", db)
    except Exception as e:
        logger.error(f"Error in /predict-webcam: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode webcam base64 image data."
        )


@router.get("/history", response_model=List[PredictionHistoryResponse])
def get_history(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """
    Retrieve logs from database, sorted by date.
    """
    return db_service.get_history(db, skip=skip, limit=limit)


@router.delete("/history/{history_id}", status_code=status.HTTP_200_OK)
def delete_history_item(history_id: int, db: Session = Depends(get_db)):
    """
    Delete a specific history log item.
    """
    deleted = db_service.delete_history_item(db, history_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History log item with ID {history_id} not found."
        )
    return {"message": f"Successfully deleted history item {history_id}."}


@router.get("/dashboard", response_model=DashboardStats)
def get_dashboard(db: Session = Depends(get_db)):
    """
    Retrieve statistics aggregated for the dashboard.
    """
    return dashboard_service.get_dashboard_statistics(db)
