import os
import contextlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.core.config import settings
from backend.app.core.database import engine, Base
from backend.app.core.logging import logger
from backend.app.api.endpoints import router as api_router
from backend.app.ml.initialize_models import initialize_ml_models

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown lifecycles.
    """
    logger.info("--- Starting FastAPI Server lifecycles ---")
    
    # 1. Initialize DB tables
    try:
        logger.info("Verifying and creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables set up successfully.")
    except Exception as e:
        logger.error(f"Error initializing database tables: {str(e)}")
        logger.warning("If PostgreSQL is not running yet, it will reconnect on queries.")
        
    # 2. Initialize baseline ML Models
    try:
        logger.info("Verifying and initializing ML Models...")
        initialize_ml_models()
        logger.info("Models initialized and loaded into memory.")
    except Exception as e:
        logger.error(f"Error during ML model initialization: {str(e)}")

    yield
    
    logger.info("--- Stopping FastAPI Server lifecycles ---")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS Middleware Configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount API Router
app.include_router(api_router, prefix="/api")

# 2. Mount Uploads folder as static to serve predicted images
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# 3. Mount Frontend static files
# To serve index.html properly on root access, we place it last.
if os.path.exists(settings.STATIC_DIR):
    app.mount("/", StaticFiles(directory=settings.STATIC_DIR, html=True), name="static")
else:
    logger.warning(f"Static directory '{settings.STATIC_DIR}' does not exist yet. Please create it.")
    
@app.exception_handler(404)
async def custom_404_handler(request, exc):
    """
    Fallback 404 handler to serve index.html for SPA client-side routing.
    """
    index_path = os.path.join(settings.STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "Not Found"}
