# ============================================================
# TASK 4: Disease Prediction from Medical Data — CodeAlpha ML
# ===========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("  Note: XGBoost not installed. Run: pip install xgboost")

print("=" * 60)
print("  TASK 4: DISEASE PREDICTION FROM MEDICAL DATA")
print("=" * 60)

# ─────────────────────────────────────────────
# HELPER: LOAD HEART DISEASE DATA (UCI)
# ─────────────────────────────────────────────
def load_heart_disease():
    """
    Load Heart Disease dataset from UCI repository.
    Falls back to sklearn's breast cancer if unavailable.
    """
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                "thalach","exang","oldpeak","slope","ca","thal","target"]
        df = pd.read_csv(url, names=cols, na_values="?")
        df.dropna(inplace=True)
        df["target"] = (df["target"] > 0).astype(int)  # Binary: disease or not
        X = df.drop("target", axis=1)
        y = df["target"]
        feature_names = cols[:-1]
        dataset_name  = "Heart Disease (UCI)"
    except Exception:
        print("  Could not load Heart Disease dataset. Using Breast Cancer instead.")
        bc = load_breast_cancer()
        X  = pd.DataFrame(bc.data, columns=bc.feature_names)
        y  = pd.Series(bc.target)
        feature_names = list(bc.feature_names)
        dataset_name  = "Breast Cancer (sklearn)"
    return X, y, feature_names, dataset_name


def load_diabetes_pima():
    """
    Load Pima Indians Diabetes dataset from Kaggle/local.
    Falls back to synthetic if not found.
    """
    try:
        df = pd.read_csv("diabetes.csv")  # Download from Kaggle
        X  = df.drop("Outcome", axis=1)
        y  = df["Outcome"]
        return X, y, list(X.columns), "Pima Diabetes"
    except FileNotFoundError:
        # Fallback: sklearn diabetes (regression target → binarize)
        d    = load_diabetes()
        X    = pd.DataFrame(d.data, columns=d.feature_names)
        y    = pd.Series((d.target > d.target.median()).astype(int))
        return X, y, list(d.feature_names), "Sklearn Diabetes (binarised)"


# ─────────────────────────────────────────────
# SELECT DATASET
# ─────────────────────────────────────────────
# Change this to switch datasets: "heart" | "diabetes" | "cancer"
DATASET = "heart"

print(f"\n[1/6] Loading Dataset: {DATASET}...")
if DATASET == "heart":
    X, y, feature_names, dataset_name = load_heart_disease()
elif DATASET == "diabetes":
    X, y, feature_names, dataset_name = load_diabetes_pima()
else:
    bc = load_breast_cancer()
    X  = pd.DataFrame(bc.data, columns=bc.feature_names)
    y  = pd.Series(bc.target)
    feature_names = list(bc.feature_names)
    dataset_name  = "Breast Cancer"

print(f"  Dataset     : {dataset_name}")
print(f"  Shape       : {X.shape}")
print(f"  Class balance:\n{y.value_counts()}")

# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
print("\n[2/6] Exploratory Data Analysis...")

df_plot = pd.DataFrame(X, columns=feature_names)
df_plot["target"] = y.values

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Task 4 — EDA: {dataset_name}", fontsize=14, fontweight="bold")

# Class distribution
y.value_counts().plot(kind="bar", ax=axes[0,0], color=["#27ae60","#e74c3c"], edgecolor="black")
axes[0,0].set_title("Class Distribution")
axes[0,0].set_xticklabels(["No Disease","Disease"], rotation=0)

# First 5 feature distributions by class
for i, col in enumerate(feature_names[:5]):
    r, c = divmod(i + 1, 3)
    df_plot.groupby("target")[col].plot(kind="kde", ax=axes[r,c], legend=(i==0))
    axes[r,c].set_title(f"Distribution: {col}")
    if i == 0:
        axes[r,c].legend(["No Disease","Disease"])

plt.tight_layout()
plt.savefig("task4_eda.png", dpi=150)
plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
corr = df_plot.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title(f"Task 4 — Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("task4_correlation.png", dpi=150)
plt.close()
print("  EDA charts saved → task4_eda.png | task4_correlation.png")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n[3/6] Preprocessing...")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train)
X_test_sc     = scaler.transform(X_test)

print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. TRAIN & COMPARE MULTIPLE MODELS
# ─────────────────────────────────────────────
print("\n[4/6] Training Models...")

models = {
    "Logistic Regression":   LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF Kernel)":      SVC(kernel="rbf", probability=True, random_state=42),
    "K-Nearest Neighbors":   KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":           GaussianNB(),
    "Random Forest":         RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":     GradientBoostingClassifier(n_estimators=200, random_state=42),
}

if XGB_AVAILABLE:
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=200, use_label_encoder=False,
        eval_metric="logloss", random_state=42
    )

cv  = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
res = {}

print(f"\n  {'Model':<25} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Recall':>8} {'CV-AUC':>8}")
print("  " + "-" * 65)

for name, mdl in models.items():
    mdl.fit(X_train_sc, y_train)
    y_pred  = mdl.predict(X_test_sc)
    y_proba = mdl.predict_proba(X_test_sc)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_proba)
    rec    = recall_score(y_test, y_pred)   # Most critical for medical!
    cv_auc = cross_val_score(mdl, X_train_sc, y_train, cv=cv, scoring="roc_auc").mean()

    res[name] = {"Accuracy": acc, "F1": f1, "ROC-AUC": auc,
                 "Recall": rec, "CV-AUC": cv_auc,
                 "y_pred": y_pred, "y_proba": y_proba}

    print(f"  {name:<25} {acc:>6.3f} {f1:>6.3f} {auc:>6.3f} {rec:>8.3f} {cv_auc:>8.3f}")

# ─────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING (Best Model)
# ─────────────────────────────────────────────
print("\n[5/6] Hyperparameter Tuning — Random Forest...")

param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_sc, y_train)
best_rf  = grid_search.best_estimator_
y_pred_b = best_rf.predict(X_test_sc)
y_prob_b = best_rf.predict_proba(X_test_sc)[:, 1]

print(f"  Best params : {grid_search.best_params_}")
print(f"  Best AUC    : {roc_auc_score(y_test, y_prob_b):.4f}")

# ─────────────────────────────────────────────
# 6. EVALUATION & VISUALISATION
# ─────────────────────────────────────────────
print("\n[6/6] Generating Evaluation Visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Task 4 — Disease Prediction Evaluation ({dataset_name})",
             fontsize=14, fontweight="bold")

# ROC Curves
for name, r in res.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    axes[0,0].plot(fpr, tpr, label=f"{name} ({r['ROC-AUC']:.2f})", linewidth=1.5)
axes[0,0].plot([0,1],[0,1],"k--")
axes[0,0].set_title("ROC Curves — All Models")
axes[0,0].set_xlabel("False Positive Rate")
axes[0,0].set_ylabel("True Positive Rate (Recall)")
axes[0,0].legend(fontsize=7)

# Model Comparison (Recall focus — critical in medicine)
model_names = list(res.keys())
recalls     = [res[m]["Recall"]  for m in model_names]
aucs        = [res[m]["ROC-AUC"] for m in model_names]
x_pos       = np.arange(len(model_names))
width       = 0.4

axes[0,1].bar(x_pos - width/2, recalls, width, label="Recall",  color="#e74c3c", edgecolor="black")
axes[0,1].bar(x_pos + width/2, aucs,    width, label="ROC-AUC", color="#3498db", edgecolor="black")
axes[0,1].set_title("Recall & AUC Comparison\n(Recall = critical in medical ML)")
axes[0,1].set_xticks(x_pos)
axes[0,1].set_xticklabels([m.replace(" ","\n") for m in model_names], fontsize=7)
axes[0,1].set_ylim(0.4, 1.05)
axes[0,1].legend()
axes[0,1].axhline(0.9, color="green", linestyle="--", alpha=0.5, label="Target")

# Confusion Matrix (best model = Random Forest tuned)
best_model_name = max(res, key=lambda k: res[k]["Recall"])  # Prioritise recall
cm = confusion_matrix(y_test, res[best_model_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=axes[1,0],
            xticklabels=["No Disease","Disease"],
            yticklabels=["No Disease","Disease"])
axes[1,0].set_title(f"Confusion Matrix\n({best_model_name})")
axes[1,0].set_xlabel("Predicted"); axes[1,0].set_ylabel("Actual")

# Feature Importance
rf_tuned = grid_search.best_estimator_
feat_imp = pd.Series(rf_tuned.feature_importances_, index=feature_names).nlargest(10)
feat_imp.sort_values().plot(kind="barh", ax=axes[1,1], color="#e67e22", edgecolor="black")
axes[1,1].set_title("Top 10 Feature Importances\n(Tuned Random Forest)")
axes[1,1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("task4_evaluation.png", dpi=150)
plt.close()
print("  Evaluation chart saved → task4_evaluation.png")

# ─────────────────────────────────────────────
# FINAL CLASSIFICATION REPORT
# ─────────────────────────────────────────────
print(f"\n  Best Model (by Recall): {best_model_name}")
print(classification_report(y_test, res[best_model_name]["y_pred"],
                             target_names=["No Disease","Disease"]))

print("\n  ⚠  MEDICAL NOTE: Recall (sensitivity) is the most critical metric.")
print("     A False Negative (missed disease) is far more costly than a False Positive.")

# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_disease(patient_data: dict):
    """
    Predict disease probability for a single patient.
    patient_data: dict with feature names as keys
    Example (Heart Disease):
        predict_disease({
            'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 130,
            'chol': 250, 'fbs': 0, 'restecg': 1, 'thalach': 160,
            'exang': 0, 'oldpeak': 1.5, 'slope': 2, 'ca': 0, 'thal': 2
        })
    """
    row     = np.array([patient_data[f] for f in feature_names], dtype=np.float32).reshape(1, -1)
    row_sc  = scaler.transform(row)
    proba   = best_rf.predict_proba(row_sc)[0][1]
    result  = "DISEASE DETECTED" if proba >= 0.5 else "NO DISEASE"
    risk    = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
    return {"prediction": result, "probability": f"{proba:.2%}", "risk_level": risk}

print("\n" + "=" * 60)
print("  TASK 4 COMPLETE ✓")
print("  Switch datasets by changing: DATASET = 'heart' | 'diabetes' | 'cancer'")
print("  Output files: task4_eda.png | task4_correlation.png | task4_evaluation.png")
print("=" * 60)