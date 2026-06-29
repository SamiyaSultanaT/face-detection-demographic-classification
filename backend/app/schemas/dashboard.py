from pydantic import BaseModel
from typing import Dict

class DashboardStats(BaseModel):
    """
    Pydantic schema representing the dashboard statistics JSON response.
    """
    total_detections: int
    gender_distribution: Dict[str, int]
    age_distribution: Dict[str, int]
