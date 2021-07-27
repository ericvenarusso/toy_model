import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from app.core.config import DATA_PATH, MODEL_FILE_NAME

def _load_train_data():
    return pd.read_csv(DATA_PATH)

def _split_datasets(df):
    X = df[["Age", "Sex", "Pclass",]]
    y = df[["Survived"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def _save_model(pipeline):
    joblib.dump(pipeline, MODEL_FILE_NAME)

def train():
    df = _load_train_data()
    X_train, X_test, y_train, y_test = _split_datasets(df)
    
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

    pipeline.fit(X_train, y_train) 

    _save_model(pipeline)

if __name__ == "__main__":
    train()