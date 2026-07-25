"""
Module: src/train.py
Project: Student AI Tools vs Exam Score Prediction
Description: Model training, evaluation, selection, and artifact persistence 
using multiple regression algorithms.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define constants and paths
TARGET_COL = "Grades_After_AI"
MODELS_DIR = Path("models")
CLEANED_DATA_PATH = MODELS_DIR / "clean_student_ai_dataset.csv"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.csv"


def load_artifacts() -> Tuple[pd.DataFrame, object, object]:
    """Load the cleaned dataset, encoder, and scaler from the models directory.

    Returns:
    - Tuple[pd.DataFrame, object, object]: Cleaned dataframe, target encoder, and feature scaler.
    """
    try:
        logger.info("Loading dataset and preprocessing artifacts...")
        
        if not CLEANED_DATA_PATH.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}")
        df = pd.read_csv(CLEANED_DATA_PATH)
        logger.info(f"Loaded dataset with shape: {df.shape}")

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

        return df, encoder, scaler
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        raise


def prepare_train_test_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the cleaned dataframe into training and testing feature/target sets.

    Parameters:
    - df (pd.DataFrame): Cleaned dataframe.

    Returns:
    - Tuple: X_train, X_test, y_train, y_test
    """
    try:
        logger.info(f"Splitting data into features and target ('{TARGET_COL}')...")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' missing from dataset.")

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error preparing train/test split: {e}")
        raise


def train_and_evaluate_models(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Train multiple regression models and evaluate them using R², MAE, MSE, and RMSE.

    Parameters:
    - X_train, X_test, y_train, y_test: Train/test dataset splits.

    Returns:
    - Tuple[Dict[str, object], pd.DataFrame]: Dictionary of trained models and dataframe of metrics.
    """
    try:
        logger.info("Initializing models for training...")
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        }

        metrics_list = []
        trained_models = {}

        for name, model in models.items():
            logger.info(f"Training model: {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)

            # Evaluation metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            logger.info(f"Evaluated {name} -> R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            metrics_list.append({
                "Model": name,
                "R2": r2,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
            })
            trained_models[name] = model

        metrics_df = pd.DataFrame(metrics_list)
        return trained_models, metrics_df
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
        raise


def select_best_model(metrics_df: pd.DataFrame, trained_models: Dict[str, object]) -> Tuple[str, object]:
    """Select the best model based on the highest R² score (or lowest RMSE).

    Parameters:
    - metrics_df (pd.DataFrame): DataFrame containing model performance metrics.
    - trained_models (Dict[str, object]): Dictionary of trained model objects.

    Returns:
    - Tuple[str, object]: Name of the best model and the model instance.
    """
    try:
        logger.info("Selecting the best performing model based on R² score...")
        best_row = metrics_df.loc[metrics_df["R2"].idxmax()]
        best_model_name = best_row["Model"]
        best_model = trained_models[best_model_name]

        logger.info(f"Best Model Selected: {best_model_name} with R²: {best_row['R2']:.4f}")
        return best_model_name, best_model
    except Exception as e:
        logger.error(f"Error selecting the best model: {e}")
        raise


def save_results(best_model: object, metrics_df: pd.DataFrame) -> None:
    """Save the best model and model metrics report to the models directory.

    Parameters:
    - best_model (object): Trained best model object.
    - metrics_df (pd.DataFrame): DataFrame of evaluation metrics.
    """
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        joblib.dump(best_model, BEST_MODEL_PATH)
        logger.info(f"Saved best model to {BEST_MODEL_PATH}")

        # Save metrics report
        metrics_df.to_csv(METRICS_PATH, index=False)
        logger.info(f"Saved model metrics report to {METRICS_PATH}")
    except Exception as e:
        logger.error(f"Error saving training artifacts: {e}")
        raise


def main() -> None:
    """Main execution function for the model training pipeline."""
    try:
        logger.info("Starting model training pipeline execution...")
        
        # 1. Load artifacts
        df, encoder, scaler = load_artifacts()
        
        # 2. Prepare train/test sets
        X_train, X_test, y_train, y_test = prepare_train_test_data(df)
        
        # 3. Train and evaluate models
        trained_models, metrics_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # 4. Select best model
        best_model_name, best_model = select_best_model(metrics_df, trained_models)
        
        # 5. Save best model and performance metrics
        save_results(best_model, metrics_df)
        
        logger.info("Model training pipeline executed successfully!")

    except Exception as e:
        logger.critical(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
