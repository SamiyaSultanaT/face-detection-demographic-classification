from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.app.core.config import settings
from backend.app.core.logging import logger

# Create SQLAlchemy engine with automatic SQLite fallback
db_url = settings.DATABASE_URL
connect_args = {}

# SQLite requires specific thread checks settings in FastAPI
if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

try:
    # Test connection and create engine
    if db_url.startswith("sqlite"):
        engine = create_engine(db_url, connect_args=connect_args)
    else:
        engine = create_engine(db_url, pool_pre_ping=True)
        # Verify connection immediately to trigger fallback if Postgres is down
        with engine.connect() as conn:
            pass
except Exception as e:
    logger.warning(f"PostgreSQL connection failed ({str(e)}). Falling back to local SQLite database...")
    db_url = "sqlite:///./predictions.db"
    connect_args = {"check_same_thread": False}
    engine = create_engine(db_url, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """
    Database dependency to yield sessions and automatically close them.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()
