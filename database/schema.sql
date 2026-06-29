-- Database Schema for Age and Gender Detection Project

-- Create prediction_history table
CREATE TABLE IF NOT EXISTS prediction_history (
    id SERIAL PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    predicted_gender VARCHAR(50) NOT NULL,
    predicted_age_group VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for optimizing queries on prediction_time
CREATE INDEX IF NOT EXISTS idx_prediction_time ON prediction_history(prediction_time DESC);
