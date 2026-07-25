"""
Module: src/preprocess.py
Project: Student AI Tools vs Exam Score Prediction
Description: Production-ready data preprocessing pipeline including data cleaning, 
missing value imputation, outlier capping, encoding, and scaling.
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define constants
TARGET_COL = "Grades_After_AI"
DATA_PATH = Path("student_ai_tools_vs_exam_scores.csv")
MODELS_DIR = Path("models")
CLEANED_DATA_PATH = MODELS_DIR / "clean_student_ai_dataset.csv"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load dataset from a given CSV file path.

    Parameters:
    - path (str or Path): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded dataframe.
    """
    path = Path(path)
    try:
        logger.info(f"Loading data from {path}")
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        df = pd.read_csv(path)
        logger.info(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def inspect_data(df: pd.DataFrame) -> None:
    """Inspect dataset shape, data types, and missing values.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    """
    try:
        logger.info("Inspecting dataset...")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Data types:\n{df.dtypes}")
        logger.info(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    except Exception as e:
        logger.error(f"Error during data inspection: {e}")
        raise


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe.

    Returns:
    - pd.DataFrame: Deduplicated dataframe.
    """
    try:
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        removed_count = initial_shape - df.shape[0]
        logger.info(f"Removed {removed_count} duplicate rows.")
        return df
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        raise


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by imputing numerical features with median and categorical with mode.

    Parameters:
    - df (pd.DataFrame): Input dataframe.

    Returns:
    - pd.DataFrame: Dataframe with missing values handled.
    """
    try:
        logger.info("Handling missing values...")
        df = df.copy()
        
        # Numerical columns: impute with median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed missing values in numerical column '{col}' with median: {median_val}")

        # Categorical columns: impute with mode
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Imputed missing values in categorical column '{col}' with mode: {mode_val}")

        return df
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        raise


def cap_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers in numerical columns using the Interquartile Range (IQR) method.

    Parameters:
    - df (pd.DataFrame): Input dataframe.

    Returns:
    - pd.DataFrame: Dataframe with capped outliers.
    """
    try:
        logger.info("Capping outliers using IQR method...")
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in num_cols:
            if col == TARGET_COL:
                continue  # Skip target variable capping
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers_count > 0:
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                logger.info(f"Capped {outliers_count} outliers in column '{col}'")

        return df
    except Exception as e:
        logger.error(f"Error capping outliers: {e}")
        raise


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Encode categorical features using OneHotEncoder/LabelEncoder.

    Parameters:
    - df (pd.DataFrame): Input dataframe.

    Returns:
    - Tuple[pd.DataFrame, LabelEncoder]: Encoded dataframe and target encoder (if applicable).
    """
    try:
        logger.info("Encoding features...")
        df = df.copy()
        
        target_encoder = None
        if TARGET_COL in df.columns and df[TARGET_COL].dtype == "object":
            target_encoder = LabelEncoder()
            df[TARGET_COL] = target_encoder.fit_transform(df[TARGET_COL])
            logger.info(f"Encoded target variable '{TARGET_COL}' using LabelEncoder.")

        # One-hot encode remaining categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
            logger.info(f"One-hot encoded categorical columns: {list(cat_cols)}")

        return df, target_encoder
    except Exception as e:
        logger.error(f"Error encoding features: {e}")
        raise


def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical features using StandardScaler.

    Parameters:
    - X (pd.DataFrame): Feature dataframe.

    Returns:
    - Tuple[pd.DataFrame, StandardScaler]: Scaled feature dataframe and the fitted scaler.
    """
    try:
        logger.info("Scaling features...")
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(scaled_array, columns=X.columns, index=X.index)
        logger.info("Feature scaling completed successfully.")
        return X_scaled, scaler
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        raise


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features (X) and target (y).

    Parameters:
    - df (pd.DataFrame): Processed dataframe.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).
    """
    try:
        logger.info(f"Splitting features and target ('{TARGET_COL}')...")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")
        
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        return X, y
    except Exception as e:
        logger.error(f"Error splitting features and target: {e}")
        raise


def save_processed_dataset(df: pd.DataFrame) -> None:
    """Save the processed dataset to the models directory.

    Parameters:
    - df (pd.DataFrame): Processed dataframe.
    """
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info(f"Saved processed dataset to {CLEANED_DATA_PATH}")
    except Exception as e:
        logger.error(f"Error saving processed dataset: {e}")
        raise


def save_preprocessor(preprocessor: Union[StandardScaler, LabelEncoder], filename: Union[str, Path]) -> None:
    """Save a preprocessor object using joblib.

    Parameters:
    - preprocessor: Preprocessing object to save.
    - filename (str or Path): Destination path.
    """
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, filename)
        logger.info(f"Saved preprocessor object to {filename}")
    except Exception as e:
        logger.error(f"Error saving preprocessor to {filename}: {e}")
        raise


def main() -> None:
    """Main execution function for the preprocessing pipeline."""
    try:
        logger.info("Starting preprocessing pipeline execution...")
        
        # 1. Load data
        df = load_data(DATA_PATH)
        
        # 2. Inspect data
        inspect_data(df)
        
        # 3. Remove duplicates
        df = remove_duplicates(df)
        
        # 4. Handle missing values
        df = handle_missing_values(df)
        
        # 5. Cap outliers
        df = cap_outliers_iqr(df)
        
        # 6. Encode features
        df, target_encoder = encode_features(df)
        
        # 7. Split features and target
        X, y = split_features_target(df)
        
        # 8. Scale features
        X_scaled, scaler = scale_features(X)
        
        # Recombine for saving clean dataset
        processed_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
        
        # 9. Save processed dataset and preprocessors
        save_processed_dataset(processed_df)
        save_preprocessor(scaler, SCALER_PATH)
        if target_encoder is not None:
            save_preprocessor(target_encoder, ENCODER_PATH)
            
        logger.info("Preprocessing pipeline executed successfully!")

    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
