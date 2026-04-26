import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test["id"]
test = test.drop(columns=["id"])
train = train.drop(columns=["id"], errors="ignore")

X = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"]

le_target = LabelEncoder()
y = le_target.fit_transform(y)

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        combined = pd.concat([X[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

X = X.fillna(X.mean())
test = test.fillna(test.mean())

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
test_scaled = pd.DataFrame(scaler.transform(test), columns=X.columns)

models = {
    "LightGBM": LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
}

n_classes = len(np.unique(y))
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

test_probs = {name: np.zeros((len(test_scaled), n_classes)) for name in models}
cv_scores = {name: [] for name in models}

for name, model in models.items():
    print(f"\nTraining {name}...")
    fold = 1
    for train_idx, val_idx in kf.split(X_scaled, y):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        score = balanced_accuracy_score(y_val, val_preds)
        cv_scores[name].append(score)
        print(f"  Fold {fold} score: {score:.4f}")
        fold += 1

    model.fit(X_scaled, y)
    test_probs[name] = model.predict_proba(test_scaled)
    print(f"  Average CV Score: {np.mean(cv_scores[name]):.4f}")

print("\n--- CV SCORES SUMMARY ---")
for name, scores in cv_scores.items():
    print(f"{name}: {np.mean(scores):.4f}")

final_probs = (
    test_probs["LightGBM"] * 0.4 +
    test_probs["XGBoost"] * 0.4 +
    test_probs["RandomForest"] * 0.2
)

final_preds = np.argmax(final_probs, axis=1)
final_preds = le_target.inverse_transform(final_preds)

submission = pd.DataFrame({
    "id": test_ids,
    "Irrigation_Need": final_preds
})

submission.to_csv("submission.csv", index=False)

print("\nDONE: submission.csv created")