from sqlalchemy.orm import Session
from backend.app.models.database import DBResultHistory
from backend.app.core.logging import logger

def save_prediction(
    db: Session, 
    image_name: str, 
    predicted_gender: str, 
    predicted_age_group: str, 
    confidence: float
) -> DBResultHistory:
    """
    Saves a prediction result to the history database.
    """
    try:
        db_history = DBResultHistory(
            image_name=image_name,
            predicted_gender=predicted_gender,
            predicted_age_group=predicted_age_group,
            confidence=round(confidence, 4)
        )
        db.add(db_history)
        db.commit()
        db.refresh(db_history)
        return db_history
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving prediction to DB: {str(e)}")
        raise

def get_history(db: Session, skip: int = 0, limit: int = 50):
    """
    Retrieves paginated logs from prediction history sorted by time descending.
    """
    try:
        return db.query(DBResultHistory)\
            .order_by(DBResultHistory.prediction_time.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()
    except Exception as e:
        logger.error(f"Error fetching prediction history: {str(e)}")
        return []

def delete_history_item(db: Session, history_id: int) -> bool:
    """
    Deletes a specific history record from the database by ID.
    Returns True if successfully deleted, False otherwise.
    """
    try:
        item = db.query(DBResultHistory).filter(DBResultHistory.id == history_id).first()
        if not item:
            return False
        db.delete(item)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting history item ID {history_id}: {str(e)}")
        raise
