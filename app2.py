import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
original = pd.read_csv("irrigation_prediction.csv")

print(f"Train size before merge: {len(train)}")
print(f"Original dataset size: {len(original)}")

test_ids = test["id"]
test = test.drop(columns=["id"])
train = train.drop(columns=["id"], errors="ignore")
original = original.drop(columns=["id"], errors="ignore")

common_cols = list(set(original.columns) & set(train.columns))
train = pd.concat([train[common_cols], original[common_cols]], ignore_index=True)

print(f"Train size after merge: {len(train)}")

X = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"]

feature_cols = X.columns.tolist()
test = test[feature_cols]

le_target = LabelEncoder()
y = le_target.fit_transform(y)

for col in feature_cols:
    if X[col].dtype == "object":
        le = LabelEncoder()
        combined = pd.concat([X[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

X = X.fillna(X.mean(numeric_only=True))
test = test.fillna(test.mean(numeric_only=True))

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
test_scaled = pd.DataFrame(scaler.transform(test), columns=feature_cols)

baseline_models = {
    "Decision Tree (Overfit)": DecisionTreeClassifier(random_state=42),
    "Decision Tree (Depth=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
}

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
baseline_results = {}

print("Baseline model comparison")

for name, model in baseline_models.items():
    scores = []
    print(f"\n{name}")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled, y), 1):
        X_tr, X_val = X_scaled.iloc[tr_idx], X_scaled.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        score = balanced_accuracy_score(y_val, model.predict(X_val))
        scores.append(score)

        print(f"Fold {fold}: {score:.4f}")

    avg = np.mean(scores)
    baseline_results[name] = avg
    print(f"Average: {avg:.4f}")

print("\nBaseline summary")

for name, score in sorted(baseline_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.4f}")

best_baseline = max(baseline_results, key=baseline_results.get)
print(f"\nBest baseline model: {best_baseline} ({baseline_results[best_baseline]:.4f})")

rf_check = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_check.fit(X_scaled, y)
print(confusion_matrix(y, rf_check.predict(X_scaled)))

ensemble_models = {
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
test_probs = {name: np.zeros((len(test_scaled), n_classes)) for name in ensemble_models}
cv_scores = {name: [] for name in ensemble_models}

for name, model in ensemble_models.items():
    print(f"\nTraining {name}")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled, y), 1):
        X_tr, X_val = X_scaled.iloc[tr_idx], X_scaled.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        score = balanced_accuracy_score(y_val, model.predict(X_val))
        cv_scores[name].append(score)

        print(f"Fold {fold}: {score:.4f}")

    model.fit(X_scaled, y)
    test_probs[name] = model.predict_proba(test_scaled)

    print(f"Average CV Score: {np.mean(cv_scores[name]):.4f}")

print("\nEnsemble summary")

for name, scores in cv_scores.items():
    print(f"{name}: {np.mean(scores):.4f}")

final_probs = (
    test_probs["LightGBM"] * 0.4 +
    test_probs["XGBoost"] * 0.4 +
    test_probs["RandomForest"] * 0.2
)

final_preds = le_target.inverse_transform(np.argmax(final_probs, axis=1))

submission = pd.DataFrame({"id": test_ids, "Irrigation_Need": final_preds})
submission.to_csv("submission.csv", index=False)

print("submission.csv created")