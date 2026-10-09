import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------
# Display Section
# ---------------------------------------------------------

def DisplayInfo(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

def LoadDataset(DataPath):

    DisplayInfo("STEP 1 : LOAD DATASET")

    try:
        df = pd.read_csv(DataPath)

        print("Dataset loaded successfully")
        print("\nFirst 5 records:")
        print(df.head())

        print("\nShape of dataset:")
        print(df.shape)

        print("\nColumn names:")
        print(df.columns.tolist())

        return df

    except Exception as e:

        logging.error("Error while loading dataset")
        print("Error:", e)

        return None


# ---------------------------------------------------------
# Show Dataset Information
# ---------------------------------------------------------

def ShowData(df):

    DisplayInfo("DATASET INFORMATION")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)


# ---------------------------------------------------------
# Clean Dataset
# ---------------------------------------------------------

def CleanTitanicData(df):

    DisplayInfo("STEP 2 : DATA CLEANING")

    df = df.copy()

    # ---------------------------------------------
    # Remove unwanted columns
    # ---------------------------------------------

    drop_columns = ["Passengerid", "zero"]

    existing_columns = [
        col for col in drop_columns
        if col in df.columns
    ]

    print("\nColumns removed:")
    print(existing_columns)

    df.drop(
        columns=existing_columns,
        inplace=True
    )

    print("\nDataset after removing unwanted columns:")
    print(df.head())

    print("\nShape after removing columns:")
    print(df.shape)

    # ---------------------------------------------
    # Separate target
    # ---------------------------------------------

    if "Survived" not in df.columns:

        raise ValueError(
            "Target column 'Survived' not found"
        )

    # ---------------------------------------------
    # Convert all feature columns to numeric
    # ---------------------------------------------

    feature_columns = [
        col for col in df.columns
        if col != "Survived"
    ]

    for col in feature_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    print("\nMissing values after numeric conversion:")
    print(df.isnull().sum())

    return df


# ---------------------------------------------------------
# Save Model
# ---------------------------------------------------------

def SaveModel(model, filename):

    try:

        joblib.dump(model, filename)

        print(
            "\nModel saved successfully as:",
            filename
        )

    except Exception as e:

        print("Error while saving model:", e)


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

def LoadSavedModel(filename):

    try:

        model = joblib.load(filename)

        print(
            "\nModel loaded successfully from:",
            filename
        )

        return model

    except Exception as e:

        print("Error while loading model:", e)

        return None


# ---------------------------------------------------------
# Train Titanic Model
# ---------------------------------------------------------

def TrainTitanicModel(df):

    DisplayInfo("STEP 3 : PREPARE FEATURES AND TARGET")

    # ---------------------------------------------
    # X = Features
    # Y = Target
    # ---------------------------------------------

    X = df.drop("Survived", axis=1)

    Y = df["Survived"]

    print("\nFeatures:")
    print(X.head())

    print("\nTarget:")
    print(Y.head())

    print("\nShape of X:")
    print(X.shape)

    print("\nShape of Y:")
    print(Y.shape)

    # ---------------------------------------------
    # Train Test Split
    # ---------------------------------------------

    DisplayInfo("STEP 4 : TRAIN TEST SPLIT")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    print("Y_train shape:", Y_train.shape)
    print("Y_test shape :", Y_test.shape)

    # ---------------------------------------------
    # Numeric Preprocessing Pipeline
    # ---------------------------------------------

    DisplayInfo("STEP 5 : CREATE ML PIPELINE")

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )
    ])

    # ---------------------------------------------
    # Complete ML Pipeline
    # ---------------------------------------------

    model_pipeline = Pipeline([

        (
            "preprocessor",
            numeric_pipeline
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    print("\nPipeline created successfully")

    # ---------------------------------------------
    # Train Model
    # ---------------------------------------------

    DisplayInfo("STEP 6 : MODEL TRAINING")

    model_pipeline.fit(
        X_train,
        Y_train
    )

    print("Model trained successfully")

    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    DisplayInfo("STEP 7 : MODEL PREDICTION")

    Y_pred = model_pipeline.predict(X_test)

    print("\nActual values:")
    print(Y_test.values)

    print("\nPredicted values:")
    print(Y_pred)

    # ---------------------------------------------
    # Accuracy
    # ---------------------------------------------

    DisplayInfo("STEP 8 : MODEL EVALUATION")

    accuracy = accuracy_score(
        Y_test,
        Y_pred
    )

    precision = precision_score(
        Y_test,
        Y_pred,
        zero_division=0
    )

    recall = recall_score(
        Y_test,
        Y_pred,
        zero_division=0
    )

    f1 = f1_score(
        Y_test,
        Y_pred,
        zero_division=0
    )

    print("\nAccuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # ---------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        Y_test,
        Y_pred
    )

    print(cm)

    # ---------------------------------------------
    # Classification Report
    # ---------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            Y_test,
            Y_pred,
            zero_division=0
        )
    )

    # ---------------------------------------------
    # Cross Validation
    # ---------------------------------------------

    DisplayInfo("STEP 9 : CROSS VALIDATION")

    cv_scores = cross_val_score(
        model_pipeline,
        X,
        Y,
        cv=5,
        scoring="accuracy"
    )

    print("Cross Validation Scores:")
    print(cv_scores)

    print(
        "\nMean Cross Validation Accuracy:",
        cv_scores.mean()
    )

    # ---------------------------------------------
    # Model Coefficients
    # ---------------------------------------------

    DisplayInfo("STEP 10 : MODEL COEFFICIENTS")

    logistic_model = model_pipeline.named_steps[
        "model"
    ]

    print("\nIntercept:")
    print(logistic_model.intercept_)

    print("\nCoefficients:")

    for feature, coefficient in zip(
        X.columns,
        logistic_model.coef_[0]
    ):

        print(
            feature,
            ":",
            coefficient
        )

    # ---------------------------------------------
    # Save Complete Pipeline
    # ---------------------------------------------

    DisplayInfo("STEP 11 : SAVE MODEL")

    SaveModel(
        model_pipeline,
        "marvelloustitanic.pkl"
    )

    # ---------------------------------------------
    # Load Saved Pipeline
    # ---------------------------------------------

    DisplayInfo("STEP 12 : LOAD SAVED MODEL")

    loaded_model = LoadSavedModel(
        "marvelloustitanic.pkl"
    )

    # ---------------------------------------------
    # Test Loaded Model
    # ---------------------------------------------

    if loaded_model is not None:

        loaded_prediction = loaded_model.predict(
            X_test
        )

        print("\nPrediction using loaded model:")
        print(loaded_prediction)

        loaded_accuracy = accuracy_score(
            Y_test,
            loaded_prediction
        )

        print(
            "\nLoaded model accuracy:",
            loaded_accuracy
        )

    return model_pipeline


# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------

def MarvellousTitanicLogistics(DataPath):

    try:

        # Step 1
        df = LoadDataset(DataPath)

        if df is None:
            return

        # Show information
        ShowData(df)

        # Step 2
        df = CleanTitanicData(df)

        # Step 3 onwards
        TrainTitanicModel(df)

    except Exception as e:

        logging.error(
            "Application failed"
        )

        print(
            "\nError:",
            e
        )


# ---------------------------------------------------------
# Program Execution
# ---------------------------------------------------------

def main():

    MarvellousTitanicLogistics(
        "MarvellousTitanicDataset.csv"
    )


if __name__ == "__main__":
    main()