"""
Week 2 Task - Supervised Machine Learning Models (Scikit-learn)
===============================================================
Algorithms : Linear Regression, Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors
Regression : predict Annual_Income (a number)
Classification : predict Approved (loan approved: 1 / rejected: 0)
Data       : data/loan_applicants_clean.csv (output of the Week 1 preprocessing task)

Run from the repository root:   python3 src/train_models.py
Creates: results/*.csv, results/charts/*.png, results/log.json, models/*.joblib
"""
import io
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

OUT = "results"
os.makedirs(f"{OUT}/charts", exist_ok=True)
os.makedirs("models", exist_ok=True)
pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 30)


# ===== SETUP: capture printed output per step (used for the PDF report) =====
class Tee:
    def __init__(self):
        self.buf = io.StringIO()

    def write(self, s):
        self.buf.write(s)
        sys.__stdout__.write(s)

    def flush(self):
        sys.__stdout__.flush()


tee = Tee()
sys.stdout = tee
LOG, _current = {}, None


def step(n):
    global _current
    if _current is not None:
        LOG[str(_current)] = tee.buf.getvalue().rstrip()
    tee.buf = io.StringIO()
    _current = n


# ===== SETUP: chart style and plotting helpers =====
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e2e1dc", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 170,
})


def save(fig, name):
    fig.savefig(f"{OUT}/charts/{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_regression(results, y_true, y_pred, best):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    r = results["R2"].sort_values()
    axes[0].barh(list(r.index), r.values, color=BLUE, height=0.55)
    for y, v in enumerate(r.values):
        axes[0].text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("R2 on the test set (higher is better)")
    axes[0].set_title("Regression: model comparison")
    axes[0].grid(axis="y", visible=False)
    axes[0].tick_params(axis="y", labelsize=7.5)
    axes[1].scatter(y_true, y_pred, s=14, color=BLUE, alpha=0.65, edgecolor=SURFACE, linewidth=0.4)
    lim = [min(y_true.min(), y_pred.min()) - 0.2, max(y_true.max(), y_pred.max()) + 0.2]
    axes[1].plot(lim, lim, color=ORANGE, linewidth=1.5, linestyle="--")
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("Actual income (z-score)")
    axes[1].set_ylabel("Predicted income (z-score)")
    axes[1].set_title(f"{best}: predicted vs actual")
    fig.tight_layout()
    save(fig, "01_regression_comparison")


def plot_metrics(results, name, title):
    metrics = ["Accuracy", "F1", "ROC_AUC"]
    colors_ = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(results))
    w = 0.26
    for i, (m, c) in enumerate(zip(metrics, colors_)):
        bars = ax.bar(x + (i - 1) * w, results[m], w, color=c, label=m.replace("_", "-"))
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.2f}",
                    ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(results.index, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (test set)")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, name)


def plot_roc(probas, y_true, aucs):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for (name, p), c in zip(probas.items(), [BLUE, ORANGE, AQUA, YELLOW]):
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, color=c, linewidth=2, label=f"{name} (AUC {aucs[name]:.2f})")
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves of the tuned classifiers")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "03_roc_curves")


def plot_confusion(preds, y_true):
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, BLUE])
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))
    for ax, (name, p) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, p)
        ax.imshow(cm, cmap=cmap)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > 0.7 * cm.max() else INK)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Rejected", "Approved"], fontsize=7)
        ax.set_yticklabels(["Rejected", "Approved"], fontsize=7)
        ax.set_xlabel("Predicted", fontsize=7.5)
        ax.set_title(name, fontsize=8)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Actual", fontsize=7.5)
    fig.tight_layout()
    save(fig, "04_confusion_matrices")


def plot_importance(imp):
    imp = imp.sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(list(imp.index), imp.values, color=BLUE, height=0.55)
    for y, v in enumerate(imp.values):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, imp.max() * 1.15)
    ax.set_xlabel("Importance (share of total)")
    ax.set_title("Random Forest: top 10 features for predicting approval")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "05_feature_importance")


# ===== STEP 1: Load the data and define the two prediction problems =====
step(1)
df = pd.read_csv("data/loan_applicants_clean.csv")
print("Shape:", df.shape)
print("All columns numeric:", all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes))

# Regression problem: predict Annual_Income (a continuous number, stored as a z-score)
X_reg, y_reg = df.drop(columns=["Annual_Income", "Approved"]), df["Annual_Income"]
# Classification problem: predict Approved (0 = rejected, 1 = approved)
X_clf, y_clf = df.drop(columns=["Approved"]), df["Approved"]
# Standard deviation of Annual_Income before scaling (Week 1, Step 11) - converts errors back to Rs.
INCOME_STD = 227053.6

print("\nRegression features :", list(X_reg.columns))
print("Regression target   : Annual_Income")
print("\nClassification target balance (%):")
print((y_clf.value_counts(normalize=True) * 100).round(1))

# ===== STEP 2: Split into training and test sets =====
step(2)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)   # stratify keeps the class balance
print("Regression     - train:", Xr_train.shape, " test:", Xr_test.shape)
print("Classification - train:", Xc_train.shape, " test:", Xc_test.shape)
print("Approval rate in train / test (%):",
      round(yc_train.mean() * 100, 1), "/", round(yc_test.mean() * 100, 1))

# ===== STEP 3: Regression models - Linear Regression, KNN, Decision Tree, Random Forest =====
step(3)
reg_models = {
    "Linear Regression": LinearRegression(),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=7),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rows, reg_pred = [], {}
for name, model in reg_models.items():
    model.fit(Xr_train, yr_train)                       # train
    pred = model.predict(Xr_test)                       # predict on unseen data
    reg_pred[name] = pred
    mae = mean_absolute_error(yr_test, pred)
    rows.append({"Model": name, "MAE": mae,
                 "RMSE": np.sqrt(mean_squared_error(yr_test, pred)),
                 "R2": r2_score(yr_test, pred),
                 "CV_R2": cross_val_score(model, Xr_train, yr_train, cv=kf, scoring="r2").mean(),
                 "MAE_Rs": mae * INCOME_STD})
reg_results = pd.DataFrame(rows).set_index("Model")
print(reg_results.sort_values("R2", ascending=False).round({"MAE": 3, "RMSE": 3, "R2": 3, "CV_R2": 3, "MAE_Rs": 0}))
print("\nLinear Regression coefficients (effect of each feature on income, in z-score units):")
print(pd.Series(reg_models["Linear Regression"].coef_, index=X_reg.columns).sort_values().round(3))
best_reg = reg_results["CV_R2"].idxmax()        # chosen by cross-validation, not by the test set
plot_regression(reg_results, yr_test, reg_pred[best_reg], best_reg)
reg_results.to_csv(f"{OUT}/regression_results.csv")

# ===== STEP 4: Classification models - Logistic Regression, KNN, Decision Tree, Random Forest =====
step(4)
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, model in clf_models.items():
    model.fit(Xc_train, yc_train)
    pred = model.predict(Xc_test)
    proba = model.predict_proba(Xc_test)[:, 1]          # probability of "approved"
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(yc_test, pred),
                 "Precision": precision_score(yc_test, pred),
                 "Recall": recall_score(yc_test, pred),
                 "F1": f1_score(yc_test, pred),
                 "ROC_AUC": roc_auc_score(yc_test, proba),
                 "CV_Accuracy": cross_val_score(model, Xc_train, yc_train, cv=skf, scoring="accuracy").mean()})
clf_results = pd.DataFrame(rows).set_index("Model")
print(clf_results.sort_values("ROC_AUC", ascending=False).round(3))
plot_metrics(clf_results, "02_classification_metrics", "Classification: default models on the test set")
clf_results.to_csv(f"{OUT}/classification_results_default.csv")

# ===== STEP 5: Hyperparameter tuning with GridSearchCV =====
step(5)
grids = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 11, 15, 21]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42),
                      {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [1, 5, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
                      {"n_estimators": [100, 200], "max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 3]}),
}
tuned, tuned_pred, tuned_proba, tuned_rows = {}, {}, {}, []
for name, (model, grid) in grids.items():
    search = GridSearchCV(model, grid, cv=skf, scoring="roc_auc")   # 5-fold CV on the training data only
    search.fit(Xc_train, yc_train)
    best = search.best_estimator_
    tuned[name] = best
    tuned_pred[name] = best.predict(Xc_test)
    tuned_proba[name] = best.predict_proba(Xc_test)[:, 1]
    tuned_rows.append({"Model": name, "Best_parameters": str(search.best_params_),
                       "CV_AUC": search.best_score_,
                       "Accuracy": accuracy_score(yc_test, tuned_pred[name]),
                       "F1": f1_score(yc_test, tuned_pred[name]),
                       "ROC_AUC": roc_auc_score(yc_test, tuned_proba[name])})
tuned_results = pd.DataFrame(tuned_rows).set_index("Model")
print(tuned_results[["Best_parameters"]].to_string())
print()
print(tuned_results.drop(columns="Best_parameters").sort_values("CV_AUC", ascending=False).round(3))
tuned_results.to_csv(f"{OUT}/classification_results_tuned.csv")

# ===== STEP 6: Compare the models and pick the best one =====
step(6)
best_clf = tuned_results["CV_AUC"].idxmax()             # chosen by cross-validation, not by the test set
comparison = pd.DataFrame({
    "Default_AUC": clf_results["ROC_AUC"], "Tuned_AUC": tuned_results["ROC_AUC"],
    "Default_Acc": clf_results["Accuracy"], "Tuned_Acc": tuned_results["Accuracy"]})
print(comparison.round(3))
print("\nBest classifier (highest cross-validated AUC):", best_clf)
print("Best regressor  (highest cross-validated R2) :", best_reg)
final = tuned[best_clf]
print("\nConfusion matrix of", best_clf, "(rows = actual, columns = predicted):")
print(pd.DataFrame(confusion_matrix(yc_test, tuned_pred[best_clf]),
                   index=["Rejected", "Approved"], columns=["Rejected", "Approved"]))
rf = tuned["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=X_clf.columns).sort_values(ascending=False)
print("\nRandom Forest feature importance (top 6):")
print(importance.head(6).round(3))
plot_roc(tuned_proba, yc_test, tuned_results["ROC_AUC"])
plot_confusion(tuned_pred, yc_test)
plot_importance(importance)
joblib.dump(final, "models/best_classifier.joblib")
joblib.dump(reg_models[best_reg], "models/best_regressor.joblib")
print("\nSaved models/best_classifier.joblib and models/best_regressor.joblib")
# ===== END =====

step("end")
RESULTS = {
    "log": LOG,
    "best_clf": best_clf, "best_reg": best_reg,
    "reg": json.loads(reg_results.reset_index().to_json(orient="records")),
    "clf_default": json.loads(clf_results.reset_index().to_json(orient="records")),
    "clf_tuned": json.loads(tuned_results.reset_index().to_json(orient="records")),
    "n_train": int(len(Xc_train)), "n_test": int(len(Xc_test)),
    "importance_top": json.loads(importance.head(10).to_json()),
}
with open(f"{OUT}/log.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
sys.stdout = sys.__stdout__
print("Training finished. Outputs in", OUT)
