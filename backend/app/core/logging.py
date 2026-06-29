import logging
import sys
from pathlib import Path

# Create log directory if it doesn't exist
log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    ]
)

logger = logging.getLogger("age_gender_detection")
