import joblib

def predict(data):
    model = joblib.load(f"models/titanic.pkl")
    return model.predict(data)

if __name__ == "__main__":
    predict()
