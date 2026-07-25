"""
Module: src/predict.py
Project: Student AI Tools vs Exam Score Prediction
Description: Inference pipeline to load saved models and artifacts, collect user input, 
preprocess the data, and predict the student's exam score after using AI tools.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


def load_model() -> tuple:
    """Load the best trained model, feature scaler, and target encoder from disk.

    Returns:
    - tuple: (best_model, scaler, encoder)
    """
    try:
        logger.info("Loading model and artifacts...")
        
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Best model not found at {BEST_MODEL_PATH}")
        best_model = joblib.load(BEST_MODEL_PATH)
        logger.info(f"Loaded best model from {BEST_MODEL_PATH}")

        scaler = None
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"Loaded scaler from {SCALER_PATH}")
        else:
            logger.warning(f"Scaler not found at {SCALER_PATH}")

        encoder = None
        if ENCODER_PATH.exists():
            encoder = joblib.load(ENCODER_PATH)
            logger.info(f"Loaded encoder from {ENCODER_PATH}")
        else:
            logger.info(f"No encoder found at {ENCODER_PATH} (optional)")

        return best_model, scaler, encoder
    except Exception as e:
        logger.error(f"Error loading model or artifacts: {e}")
        raise


def preprocess_input(input_data: Dict[str, Any], scaler: Any = None, encoder: Any = None) -> pd.DataFrame:
    """Convert user input dictionary to a DataFrame, encode categorical features, and scale numerical columns.

    Parameters:
    - input_data (Dict[str, Any]): Raw user inputs.
    - scaler (Any): Fitted StandardScaler object.
    - encoder (Any): Fitted LabelEncoder or encoding artifacts.

    Returns:
    - pd.DataFrame: Preprocessed feature set ready for prediction.
    """
    try:
        logger.info("Preprocessing user input...")
        # Convert dictionary to single-row DataFrame
        df = pd.DataFrame([input_data])

        # Handle categorical encoding (One-hot encoding matching training setup)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

        # Ensure the feature columns align with what the scaler/model expects if training columns metadata is saved,
        # but since we process dynamically, we apply the scaler to the available numerical columns.
        if scaler is not None:
            # Reindex or align columns if necessary, or scale numerical columns present
            num_cols = df.select_dtypes(include=[np.number]).columns
            # If the scaler expects specific feature dimensions from training, handle appropriately:
            # Here we apply the scaler transform to the matching feature subset or full dataframe if columns match.
            try:
                scaled_array = scaler.transform(df)
                df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
            except ValueError as ve:
                logger.warning(f"Column mismatch during scaling, attempting alignment: {ve}")
                # Fallback alignment if schema differs slightly from inference input
                missing_cols = set(scaler.feature_names_in_) - set(df.columns)
                for col in missing_cols:
                    df[col] = 0
                df = df[scaler.feature_names_in_]
                scaled_array = scaler.transform(df)
                df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

        logger.info("Input preprocessing completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error during input preprocessing: {e}")
        raise


def predict_grade(model: Any, input_df: pd.DataFrame) -> float:
    """Predict the student's grade after AI tool usage.

    Parameters:
    - model (Any): Trained regression model.
    - input_df (pd.DataFrame): Preprocessed input features.

    Returns:
    - float: Predicted grade rounded to two decimal places.
    """
    try:
        logger.info("Making prediction...")
        prediction = model.predict(input_df)[0]
        rounded_prediction = round(float(prediction), 2)
        logger.info(f"Prediction computed: {rounded_prediction}")
        return rounded_prediction
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise


def collect_user_input() -> Dict[str, Any]:
    """Interactively collect required features from the user via the command line.

    Returns:
    - Dict[str, Any]: Dictionary containing user input values.
    """
    print("\n--- Student AI Tools & Exam Score Prediction ---")
    print("Please provide the following details:")
    
    try:
        age = float(input("Age: "))
        education_level = input("Education_Level (e.g., High School, Undergraduate): ").strip()
        study_hours = float(input("Study_Hours_Per_Day: "))
        uses_ai = input("Uses_AI (Yes/No or True/False): ").strip()
        ai_tools_used = input("AI_Tools_Used (e.g., ChatGPT, Copilot): ").strip()
        purpose_of_ai = input("Purpose_of_AI (e.g., Research, Coding, Writing): ").strip()
        grades_before_ai = float(input("Grades_Before_AI: "))
        screen_time = float(input("Daily_Screen_Time_Hours: "))

        user_data = {
            "Age": age,
            "Education_Level": education_level,
            "Study_Hours_Per_Day": study_hours,
            "Uses_AI": uses_ai,
            "AI_Tools_Used": ai_tools_used,
            "Purpose_of_AI": purpose_of_ai,
            "Grades_Before_AI": grades_before_ai,
            "Daily_Screen_Time_Hours": screen_time,
        }
        return user_data
    except Exception as e:
        logger.error(f"Invalid input format provided by user: {e}")
        raise ValueError("Invalid input format. Please enter numeric values where expected.")


def main() -> None:
    """Main execution function for the prediction pipeline."""
    try:
        logger.info("Starting prediction pipeline execution...")
        
        # 1. Load model and artifacts
        model, scaler, encoder = load_model()
        
        # 2. Collect inputs from user
        raw_input = collect_user_input()
        
        # 3. Preprocess input data
        processed_input = preprocess_input(raw_input, scaler=scaler, encoder=encoder)
        
        # 4. Predict grade
        predicted_score = predict_grade(model, processed_input)
        
        # 5. Output result
        print("\n==============================================")
        print(f"Predicted Grade After AI: {predicted_score}")
        print("==============================================")
        
        logger.info("Prediction pipeline executed successfully!")

    except Exception as e:
        logger.critical(f"Prediction pipeline failed: {e}")
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
