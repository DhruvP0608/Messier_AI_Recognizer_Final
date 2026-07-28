"""Test SMOTE + class balancing"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Create imbalanced synthetic data
print("Creating synthetic imbalanced dataset...")
X, y = make_classification(
    n_samples=5000,
    n_features=235,
    n_classes=7,
    weights=[0.39, 0.05, 0.18, 0.18, 0.05, 0.08, 0.07],
    random_state=42,
    n_informative=50,
)

#Split
split_idx = int(0.5 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Approach 1: No balancing (baseline)
print("\n1. BASELINE - No balancing:")
model1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=5000, random_state=42, n_jobs=-1)),
])
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
macro_f1_1 = f1_score(y_test, y_pred1, average="macro")
print(f"Macro F1: {macro_f1_1*100:.2f}%")

# Approach 2: Class weights only
print("\n2. CLASS WEIGHTS only:")
model2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=5000, class_weight="balanced", random_state=42, n_jobs=-1)),
])
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
macro_f1_2 = f1_score(y_test, y_pred2, average="macro")
print(f"Macro F1: {macro_f1_2*100:.2f}% ({macro_f1_2 - macro_f1_1:+.2%} vs baseline)")

# Approach 3: SMOTE + class weights (THE NEW APPROACH)
print("\n3. SMOTE + CLASS WEIGHTS (NEW):")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"  Samples: {len(X_train)} -> {len(X_train_balanced)} (SMOTE oversampling)")

model3 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=5000, class_weight="balanced", random_state=42, n_jobs=-1)),
])
model3.fit(X_train_balanced, y_train_balanced)
y_pred3 = model3.predict(X_test)
macro_f1_3 = f1_score(y_test, y_pred3, average="macro")
print(f"Macro F1: {macro_f1_3*100:.2f}% ({macro_f1_3 - macro_f1_1:+.2%} vs baseline)")

print("\n" + "="*60)
print(f"IMPROVEMENT with SMOTE + Class Weights: {(macro_f1_3 - macro_f1_1)*100:+.2f}%")
print("="*60)

# Show per-class improvements
print("\nPer-class Recall (NEW approach vs baseline):")
report_baseline = classification_report(y_test, y_pred1, output_dict=True)
report_smote = classification_report(y_test, y_pred3, output_dict=True)

for cls in sorted(set(y_train.tolist())):
    cls_str = str(cls)
    baseline_recall = report_baseline.get(cls_str, {}).get("recall", 0) * 100
    smote_recall = report_smote.get(cls_str, {}).get("recall", 0) * 100
    improvement = smote_recall - baseline_recall
    print(f"  Class {cls}: {baseline_recall:6.2f}% -> {smote_recall:6.2f}% ({improvement:+6.2f}%)")
