from pathlib import Path
import logging

import pandas as pd
import bentoml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO)


def train_model(processed_dir: Path) -> None:
    """Train the admission prediction model and store it in BentoML."""

    # Load datasets
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv")

    # Define pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    logging.info("Model trained successfully")
    logging.info("Evaluation metrics: R2=%.3f RMSE=%.3f", r2, rmse)

    # Save model to BentoML Model Store
    bento_model = bentoml.sklearn.save_model(
        "admission_lr:1.0.0",
        pipeline,
        signatures={"predict": {"batchable": True}},
    )

    logging.info("Model saved to BentoML Model Store: %s", bento_model)


def run() -> None:
    """Entry point for the training pipeline."""

    project_path = Path(__file__).resolve().parents[1]
    processed_dir = project_path / "data" / "processed"

    train_model(processed_dir)


if __name__ == "__main__":
    run()
