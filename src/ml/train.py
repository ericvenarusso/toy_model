import os

import pandas as pd

import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from category_encoders.binary import BinaryEncoder

def train():
    df = pd.read_csv("data/titanic.csv")

    X = df[["Pclass", "Sex", "Fare",]]
    y = df[["Survived"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    pipeline = Pipeline(
        steps = [
            ("Binary Enconder", BinaryEncoder(cols = ["Sex"])),
            ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                                    max_depth=5, random_state=42))
        ]
    )

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "models/titanic.pkl")

if __name__ == "__main__":
    train()
