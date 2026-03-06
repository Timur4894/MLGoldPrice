import joblib
from Model.gold_features import build_dataset


def predict_gold_30_days():
    dataset = build_dataset()

    model = joblib.load("../targets/gold_model_target_30d.pkl")
    features = joblib.load("../targets/features.pkl")

    X_new = dataset[features].iloc[-1:]

    prediction = model.predict(X_new)
    return int(prediction)


