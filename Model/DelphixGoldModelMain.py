from sklearn.metrics import accuracy_score
import joblib
import lightgbm as lgb

from gold_features import build_dataset


dataset = build_dataset()

# --------- Features & Target ---------
targets = ["target_1d", "target_10d", "target_30d"]
X = dataset.drop(targets, axis=1)

# сохраняем единый список признаков для всех моделей
joblib.dump(X.columns, "../targets/features.pkl")

# --------- Walk-forward validation ---------
initial_train_size = int(len(X) * 0.8)
test_size = 15

for target in targets:
    y = dataset[target]
    predictions = []
    y_tests = []

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.05,
        n_jobs=-1,
    )

    for end_train in range(initial_train_size, len(X) - test_size + 1, test_size):
        X_train = X.iloc[:end_train]
        y_train = y.iloc[:end_train]
        X_test = X.iloc[end_train : end_train + test_size]
        y_test = y.iloc[end_train : end_train + test_size]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        predictions.extend(pred)
        y_tests.extend(y_test)

    accuracy = accuracy_score(y_tests, predictions)
    print(f"Walk-forward accuracy for {target}:", accuracy)

    joblib.dump(model, f"gold_model_{target}.pkl")
    print(f"Model for {target} saved")
