import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Age and Gender Detection API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # Environment
    ENV: str = os.getenv("ENV", "development")
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/age_gender_detection"
    )
    
    # Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    APP_DIR: Path = Path(__file__).resolve().parent.parent
    
    UPLOAD_DIR: Path = APP_DIR / "uploads"
    MODEL_DIR: Path = APP_DIR / "ml" / "models"
    STATIC_DIR: Path = APP_DIR / "static"
    
    # ML Models config
    GENDER_MODEL_PATH: Path = MODEL_DIR / "gender_model.h5"
    AGE_MODEL_PATH: Path = MODEL_DIR / "age_model.h5"
    
    class Config:
        case_sensitive = True

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.STATIC_DIR, exist_ok=True)
