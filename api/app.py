"""
Module: api/app.py
Project: Student AI Tools vs Exam Score Prediction API
Description: Production-ready FastAPI application for serving student exam score predictions 
using pre-trained machine learning models and preprocessing artifacts.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define file paths
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Student AI Tools vs Exam Score Prediction API",
    description="Predict students' exam scores after using AI learning tools.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global variables for model artifacts
model = None
scaler = None
encoder = None


def load_artifacts() -> None:
    """Load the machine learning model, feature scaler, and target encoder into memory at startup."""
    global model, scaler, encoder
    try:
        logger.info("Loading model and artifacts at startup...")
        
        if BEST_MODEL_PATH.exists():
            model = joblib.load(BEST_MODEL_PATH)
            logger.info(f"Loaded best model from {BEST_MODEL_PATH}")
        else:
            logger.warning(f"Best model not found at {BEST_MODEL_PATH}")

        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"Loaded scaler from {SCALER_PATH}")
        else:
            logger.warning(f"Scaler not found at {SCALER_PATH}")

        if ENCODER_PATH.exists():
            encoder = joblib.load(ENCODER_PATH)
            logger.info(f"Loaded encoder from {ENCODER_PATH}")
        else:
            logger.info(f"No encoder found at {ENCODER_PATH} (optional)")

    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise


@app.on_event("startup")
def startup_event() -> None:
    """Execute startup procedures to load artifacts."""
    load_artifacts()


class StudentInput(BaseModel):
    """Pydantic model representing input features for prediction."""

    Age: int = Field(..., example=20, description="Age of the student")
    Education_Level: str = Field(..., example="Undergraduate", description="Current education level")
    Study_Hours_Per_Day: float = Field(..., example=4.5, description="Average study hours per day")
    Uses_AI: str = Field(..., example="Yes", description="Whether the student uses AI tools")
    AI_Tools_Used: str = Field(..., example="ChatGPT", description="Primary AI tools used")
    Purpose_of_AI: str = Field(..., example="Research", description="Main purpose of using AI tools")
    Grades_Before_AI: float = Field(..., example=78.5, description="Grades before implementing AI tools")
    Daily_Screen_Time_Hours: float = Field(..., example=6.0, description="Total daily screen time in hours")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 20,
                "Education_Level": "Undergraduate",
                "Study_Hours_Per_Day": 4.5,
                "Uses_AI": "Yes",
                "AI_Tools_Used": "ChatGPT",
                "Purpose_of_AI": "Research",
                "Grades_Before_AI": 78.5,
                "Daily_Screen_Time_Hours": 6.0,
            }
        }


@app.get("/", tags=["General"], summary="Root Endpoint")
def read_root() -> Dict[str, str]:
    """Root endpoint to check API status."""
    return {
        "message": "Student AI Tools vs Exam Score Prediction API",
        "status": "Running",
    }


@app.get("/health", tags=["General"], summary="Health Check Endpoint")
def health_check() -> Dict[str, str]:
    """Health check endpoint to ensure service availability."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return {"status": "healthy"}


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict Exam Score After AI Usage",
    response_description="Returns the predicted grade rounded to two decimal places.",
)
def predict_score(data: StudentInput) -> Dict[str, float]:
    """Process incoming student attributes, transform features, and return the predicted exam score."""
    try:
        logger.info("Received prediction request...")
        
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction model is not initialized.",
            )

        # Convert request body to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Apply one-hot encoding matching training pipeline
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

        # Apply scaler if available
        if scaler is not None:
            try:
                scaled_array = scaler.transform(df)
                df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
            except ValueError:
                # Align columns if inference features differ slightly from training expectations
                missing_cols = set(scaler.feature_names_in_) - set(df.columns)
                for col in missing_cols:
                    df[col] = 0
                df = df[scaler.feature_names_in_]
                scaled_array = scaler.transform(df)
                df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

        # Make prediction
        prediction = model.predict(df)[0]
        rounded_prediction = round(float(prediction), 2)

        logger.info(f"Successfully predicted grade: {rounded_prediction}")
        return {"Predicted_Grade": rounded_prediction}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(e)}",
        )
