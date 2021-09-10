import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector

from app.ml.config import DATA_PATH, MODEL_FILE_NAME


def _load_train_data() -> pd.DataFrame:
    """"
        Load the train data.
    """
    return pd.read_csv(DATA_PATH)

def _split_datasets(df: pd.DataFrame) -> np.array:
    """
        Split dataset into train and test.
    """
    X = df[["Age", "Sex", "Pclass",]]
    y = df[["Survived"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def _create_pipeline() -> Pipeline:
    """"
        Create a Sklearn pipeline that is composed of preprocessing and a model.
    """
    preprocessor = ColumnTransformer(
        transformers = [
            ("Simple Imputer", SimpleImputer(strategy="mean"), [0]),
            ("One Hot Encoding", OneHotEncoder(), make_column_selector(dtype_include="category"))
        ]
    )

    pipeline = Pipeline(
        steps = [
            ("Preprocessing", preprocessor),
            ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                                    max_depth=5, random_state=42))
        ]
    )

    return pipeline

def _save_model(pipeline: Pipeline) -> None:
    """
        Save the model pipeline into a serialized object.
    """
    joblib.dump(pipeline, MODEL_FILE_NAME)

def train() -> None:
    """
        Orchestrate the training of a machine learning model.
    """
    df = _load_train_data()
    X_train, X_test, y_train, y_test = _split_datasets(df)
    
    pipeline = _create_pipeline()
    pipeline.fit(X_train, y_train) 

    _save_model(pipeline)

if __name__ == "__main__":
    train()