import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from category_encoders.binary import BinaryEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.core.config import DATA_PATH, MODEL_FILE_NAME


def _load_train_data():
    return pd.read_csv(DATA_PATH)

def _prepare_train_data():
    df = _load_train_data()

    X = df[["Pclass", "Sex", "Fare",]]
    y = df[["Survived"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def _save_model(pipeline):
    joblib.dump(pipeline, MODEL_FILE_NAME)

def train():
    X_train, X_test, y_train, y_test = _prepare_train_data()

    pipeline = Pipeline(
        steps = [
            ("Binary Enconder", BinaryEncoder(cols = ["Sex"])),
            ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                                    max_depth=5, random_state=42))
        ]
    )

    pipeline.fit(X_train, y_train) 

    _save_model(pipeline)

if __name__ == "__main__":
    train()