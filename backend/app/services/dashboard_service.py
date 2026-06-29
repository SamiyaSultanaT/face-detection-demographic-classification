from sqlalchemy import func
from sqlalchemy.orm import Session
from backend.app.models.database import DBResultHistory
from backend.app.ml.predictor import AGE_GROUPS

def get_dashboard_statistics(db: Session):
    """
    Computes dashboard analytics from the database.
    Pre-populates values with zero to ensure charts look correct on load.
    """
    # 1. Total detections
    total_detections = db.query(func.count(DBResultHistory.id)).scalar() or 0

    # 2. Gender distribution
    # Initialize defaults
    gender_distribution = {"Male": 0, "Female": 0}
    gender_stats = (
        db.query(DBResultHistory.predicted_gender, func.count(DBResultHistory.id))
        .group_by(DBResultHistory.predicted_gender)
        .all()
    )
    for gender, count in gender_stats:
        if gender in gender_distribution:
            gender_distribution[gender] = count

    # 3. Age group distribution
    # Initialize defaults
    age_distribution = {group: 0 for group in AGE_GROUPS}
    age_stats = (
        db.query(DBResultHistory.predicted_age_group, func.count(DBResultHistory.id))
        .group_by(DBResultHistory.predicted_age_group)
        .all()
    )
    for age_group, count in age_stats:
        if age_group in age_distribution:
            age_distribution[age_group] = count

    return {
        "total_detections": total_detections,
        "gender_distribution": gender_distribution,
        "age_distribution": age_distribution,
    }
