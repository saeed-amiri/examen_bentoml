from pathlib import Path
import logging
import re

import pandas as pd
from sklearn.model_selection import train_test_split


def to_snake(name: str) -> str:
    """Normalize a column name to lower snake_case without spaces."""
    s = name.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def prepare_data(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Load raw data, clean it, split into train/test, and save the datasets."""

    # Load dataset
    df = pd.read_csv(input_path)

    # Normalize column names
    df.columns = [to_snake(col) for col in df.columns]

    # Drop unnecessary column
    df = df.drop(columns=["serial_no"])

    # Remove missing values
    df = df.dropna()

    # Define features and target
    X = df.drop(columns=["chance_of_admit"])
    y = df["chance_of_admit"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    logging.info("Data prepared and saved to %s", output_dir)


def run() -> None:
    """Entry point for the data preparation pipeline."""
    logging.basicConfig(level=logging.INFO)

    project_path = Path(__file__).resolve().parents[1]

    input_path = project_path / "data" / "raw" / "admission.csv"
    output_dir = project_path / "data" / "processed"

    prepare_data(input_path=input_path, output_dir=output_dir)


if __name__ == "__main__":
    run()
