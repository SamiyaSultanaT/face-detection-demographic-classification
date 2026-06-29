# Age and Gender Detection Using Machine Learning

An AI-powered final-year computer science web application that detects a person's age group and gender from an uploaded image or live webcam stream. The project utilizes **TensorFlow/Keras CNNs**, **OpenCV** for facial localization, a **FastAPI** backend, and a **PostgreSQL** relational database for logging prediction logs, all packaged using **Docker Compose** for easy cloud deployment.

---

# Key Features

* **Facial Detection**: Instantly localizes multiple human faces in a frame using OpenCV Haar Cascades.
* **Inference Pipeline**: Runs separate customized Deep CNNs to predict Gender and Age Group.
* **Dual Input Modes**: Supports drag-and-drop image files (PNG, JPG, WEBP) and live webcam snapshot captures.
* **Interactive Bounding Box Overlay**: Draws high-accuracy bounding boxes and classifications matching responsive scale coordinates.
* **Database Tracking**: Records image logs, classification categories, and confidence limits to PostgreSQL.
* **Dashboard Analytics**: Uses **Chart.js** to render real-time telemetry representing total detections, gender breakdowns, and age demographics.
* **Modular Codebase**: Organized directory split between FastAPI server, training pipelines, assets, and dockerized deployment config.

---

# Project Structure

```
Age Gender detection/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py         # FastAPI prediction and analytics endpoints
│   │   ├── core/
│   │   │   ├── config.py            # Pydantic Settings env configurations
│   │   │   ├── database.py          # SQLAlchemy SessionMaker engine
│   │   │   └── logging.py           # Logging module
│   │   ├── ml/
│   │   │   ├── face_detector.py     # OpenCV Haar Cascade face crop processor
│   │   │   ├── predictor.py         # Preprocess images and perform model predictions
│   │   │   └── initialize_models.py # Baseline model generator script
│   │   ├── models/
│   │   │   └── database.py          # SQLAlchemy prediction_history log model
│   │   ├── schemas/
│   │   │   ├── history.py           # Pydantic schemas for predictions
│   │   │   └── dashboard.py         # Pydantic dashboard statistics schemas
│   │   ├── services/
│   │   │   ├── db_service.py        # Database operations
│   │   │   └── dashboard_service.py # Analytics aggregation service
│   │   ├── static/                  # Frontend single-page app directory
│   │   │   ├── css/styles.css       # CSS custom glassmorphism style rules
│   │   │   ├── js/app.js            # SPA logic, camera stream & Chart.js scripts
│   │   │   └── index.html           # Main user interface entrypoint
│   │   ├── uploads/                 # Storage for predicted image results
│   │   └── main.py                  # API server bootstrap and startup lifespan
│   ├── ml_training/
│   │   ├── download_dataset.py      # Instructions and synthetic mockup tool
│   │   ├── preprocess.py            # Train/test partition and pipeline loaders
│   │   ├── train_gender.py          # Gender classification training script
│   │   └── train_age.py             # Age group classification training script
│   └── requirements.txt             # Python packages
│   └── Dockerfile                   # Python container image specifications
├── database/
│   └── schema.sql                   # Raw SQL schema queries
├── docker-compose.yml               # Service orchestrator (Postgres + Python)
├── .env.example                     # Environment templates guide
└── README.md                        # Project quickstart guide
```

---

# Quickstart Guide

### Option 1: Running with Docker Compose (Recommended)

Make sure you have [Docker and Docker Compose](https://docs.docker.com/get-docker/) installed.

1. **Clone or Copy** the project workspace.
2. In the project root directory, spin up the Docker containers:
   ```bash
   docker-compose up --build
   ```
3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```
   *The FastAPI server will automatically verify and initialize database tables and create baseline ML model files if they are not already trained.*

---

### Option 2: Running Locally (Without Docker)

#### Prerequisites
* Python 3.10+
* PostgreSQL running locally with database `age_gender_detection`.

#### Installation
1. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
2. **Install Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   Copy `.env.example` as `.env` and adjust your DB URL:
   ```bash
   cp ../.env.example .env
   # Edit database connection settings if necessary
   ```
4. **Run Application**:
   ```bash
   python app/main.py
   ```
   Now visit: `http://localhost:8000`.

---

# Model Training Instructions

To train models on the real **UTKFace** dataset:

1. **Download the Dataset**:
   Run the download helper inside the `ml_training` directory:
   ```bash
   cd backend/ml_training
   python download_dataset.py
   ```
   *This displays instructions on where to download the UTKFace ZIP (from Kaggle) and places it into `backend/ml_training/dataset/`.*
   *By default, the script also creates a `dataset_mock` folder containing synthetic images so you can test training commands instantly!*

2. **Train Gender Model**:
   ```bash
   python train_gender.py --dataset ./dataset_mock --epochs 5
   ```
3. **Train Age Model**:
   ```bash
   python train_age.py --dataset ./dataset_mock --epochs 5
   ```
   *The best weights (monitored by validation loss) will automatically overwrite the baseline models located in `backend/app/ml/models/`.*
