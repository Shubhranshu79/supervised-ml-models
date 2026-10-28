"""Builds report/Week2_Supervised_ML_Models_Report.pdf from the real code, outputs and charts.
Run from the repository root after train_models.py:   python3 src/build_report.py"""
import json
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (CondPageBreak, Image, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

RES = "results"
PDF = "report/Week2_Supervised_ML_Models_Report.pdf"
AUTHOR, DATE = "Shubh", "28 October 2026"

R = json.load(open(f"{RES}/log.json"))
LOG = R["log"]
reg = {r["Model"]: r for r in R["reg"]}
clf = {r["Model"]: r for r in R["clf_default"]}
tun = {r["Model"]: r for r in R["clf_tuned"]}

# ---------- real code of each step ----------
steps, titles, cur = {}, {}, None
for line in open("src/train_models.py").read().splitlines():
    m = re.match(r"# ===== STEP (\d+): (.*) =====", line)
    if m:
        cur = int(m.group(1))
        titles[cur], steps[cur] = m.group(2), []
        continue
    if line.startswith("# ===== END") or line.startswith("# ===== SETUP"):
        cur = None
        continue
    if cur is not None and not line.startswith("step("):
        steps[cur].append(line)
CODE = {k: "\n".join(v).strip("\n") for k, v in steps.items()}


def wrap(text, width=118):
    out = []
    for ln in text.splitlines():
        if len(ln) <= width:
            out.append(ln)
        else:
            indent = len(ln) - len(ln.lstrip())
            out += textwrap.wrap(ln, width, subsequent_indent=" " * (indent + 4)) or [""]
    return "\n".join(out)


# ---------- styles ----------
INK, MUTED = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e")
BLUE = colors.HexColor("#2a78d6")
HDR, LIGHT, LINE = colors.HexColor("#22324a"), colors.HexColor("#f4f5f7"), colors.HexColor("#d5d7db")
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6, leading=14,
                      textColor=INK, spaceAfter=5)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=13, bulletIndent=2, spaceAfter=2.5)
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=16, leading=20,
                    textColor=HDR, spaceBefore=4, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=12, leading=16,
                    textColor=HDR, spaceBefore=8, spaceAfter=4)
label = ParagraphStyle("label", parent=body, fontName="Helvetica-Bold", fontSize=8, leading=10,
                       textColor=MUTED, spaceBefore=5, spaceAfter=2)
caption = ParagraphStyle("caption", parent=body, fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=8)
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=11, spaceAfter=0)
cell_h = ParagraphStyle("cell_h", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
code_style = ParagraphStyle("code", fontName="Courier", fontSize=6.7, leading=8.3, textColor=INK,
                            backColor=LIGHT, borderPadding=(4, 4, 4, 4), leftIndent=4, rightIndent=4,
                            spaceBefore=3, spaceAfter=7)
out_style = ParagraphStyle("out", parent=code_style, backColor=colors.HexColor("#fbf6f2"))
title_style = ParagraphStyle("title", parent=h1, fontSize=27, leading=33, spaceAfter=6)
sub_style = ParagraphStyle("sub", parent=body, fontSize=13, leading=18, textColor=MUTED, spaceAfter=18)
W = A4[0] - 36 * mm


def P(t, s=body):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(rows, widths):
    d = [[Paragraph(str(c), cell_h if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(d, colWidths=[w * W for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HDR), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE), ("BOX", (0, 0), (-1, -1), 0.3, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [t, Spacer(1, 8)]


def image(name, cap, width=0.92):
    path = f"{RES}/charts/{name}.png"
    iw, ih = ImageReader(path).getSize()
    w = W * width
    return [Image(path, width=w, height=w * ih / iw), P(cap, caption)]


def callout(text):
    t = Table([[Paragraph(text, cell)]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eaf2fc")),
        ("LINEBEFORE", (0, 0), (0, -1), 3, BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return [t, Spacer(1, 6)]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Week 2 Task - Supervised Machine Learning Models")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


f3 = lambda v: f"{v:.3f}"
RL, LR_, RF, DT, KN = "Random Forest", "Linear Regression", "Random Forest", "Decision Tree", "KNN Regressor"

reg_rows = [["Model", "MAE", "RMSE", "R2 (test)", "R2 (cross-val.)", "Average error in Rs."]]
for n in sorted(reg, key=lambda k: -reg[k]["CV_R2"]):
    r = reg[n]
    reg_rows.append([n, f3(r["MAE"]), f3(r["RMSE"]), f3(r["R2"]), f3(r["CV_R2"]), f"{r['MAE_Rs']:,.0f}"])
clf_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV accuracy"]]
for n in sorted(clf, key=lambda k: -clf[k]["ROC_AUC"]):
    r = clf[n]
    clf_rows.append([n, f3(r["Accuracy"]), f3(r["Precision"]), f3(r["Recall"]), f3(r["F1"]),
                     f3(r["ROC_AUC"]), f3(r["CV_Accuracy"])])
tun_rows = [["Model", "Best parameters", "CV AUC", "Test accuracy", "Test F1", "Test AUC"]]
for n in sorted(tun, key=lambda k: -tun[k]["CV_AUC"]):
    r = tun[n]
    tun_rows.append([n, r["Best_parameters"], f3(r["CV_AUC"]), f3(r["Accuracy"]), f3(r["F1"]), f3(r["ROC_AUC"])])

STEP_TEXT = {
    1: ("Both tasks use the cleaned dataset from Week 1 (515 rows, every column numeric). Two prediction "
        "problems are defined. <b>Regression</b> predicts a number: Annual_Income. <b>Classification</b> predicts "
        "a category: Approved (1 = approved, 0 = rejected). The target column is removed from the features (X) "
        "so the model cannot see the answer. The features are already standardised (z-scores), so regression "
        "errors are in z-score units and are converted back to rupees with the income standard deviation from "
        "Week 1 (Rs. 227,053.6).",
        "13 features for regression and 14 for classification. 61.4% of applicants are approved, so the two "
        "classes are reasonably balanced."),
    2: ("A model must be tested on data it has never seen, otherwise the score only measures memorisation. "
        "The data is split 80% / 20% with <b>train_test_split</b>. <b>random_state=42</b> makes the split "
        "repeatable and <b>stratify</b> keeps the same approval rate in both parts of the classification split.",
        f"{R['n_train']} rows for training and {R['n_test']} rows for testing. The approval rate is "
        "61.4% in the training set and 61.2% in the test set."),
    3: ("Four regression models are trained with <b>fit()</b>, used to <b>predict()</b> on the test set, and "
        "scored with MAE, RMSE and R2 (explained in Section 3). <b>5-fold cross-validation</b> on the training "
        "data (CV R2) shows how stable each score is across different slices of the data.",
        f"Random Forest has the best test R2 ({f3(reg[RF]['R2'])}), but Linear Regression has the best "
        f"cross-validated R2 ({f3(reg[LR_]['CV_R2'])} vs {f3(reg[RF]['CV_R2'])}), so the two are practically "
        f"tied. <b>Linear Regression is chosen</b> because cross-validation is a more reliable guide than one "
        f"103-row test set, and the model is simpler. Its average error is about Rs. "
        f"{reg[LR_]['MAE_Rs']:,.0f}. Age (0.67) and Education (0.52) are the main drivers of income. KNN is "
        f"the weakest (R2 {f3(reg[KN]['R2'])}), probably because distances work poorly with many 0/1 "
        "dummy columns."),
    4: ("The four classification models are trained the same way and scored with accuracy, precision, recall, "
        "F1 and ROC-AUC, plus cross-validated accuracy. These first results use sensible default settings.",
        f"All four models beat random guessing. Random Forest has the highest accuracy "
        f"({f3(clf[RF]['Accuracy'])}) and F1 ({f3(clf[RF]['F1'])}); Logistic Regression has the highest "
        f"ROC-AUC ({f3(clf['Logistic Regression']['ROC_AUC'])}). The untuned Decision Tree is the weakest "
        f"(accuracy {f3(clf[DT]['Accuracy'])}, AUC {f3(clf[DT]['ROC_AUC'])}) because a single deep tree is "
        "unstable. The top three models are close: with only 103 test rows, one applicant equals about one "
        "percentage point."),
    5: ("Default settings are only a first guess. <b>GridSearchCV</b> tries every combination of chosen "
        "hyperparameters with 5-fold cross-validation on the <b>training data only</b> (the test set stays "
        "untouched) and keeps the combination with the best ROC-AUC. Tuned settings: C (regularisation "
        "strength) for Logistic Regression, k (number of neighbours) for KNN, depth and leaf size for the "
        "tree, and number of trees, depth and leaf size for the forest.",
        f"Tuning helped most where defaults were weak: Decision Tree AUC rose from "
        f"{f3(clf[DT]['ROC_AUC'])} to {f3(tun[DT]['ROC_AUC'])} (a shallow tree of depth 2 generalises better) "
        f"and KNN from {f3(clf['KNN']['ROC_AUC'])} to {f3(tun['KNN']['ROC_AUC'])} (k = 21 gives smoother "
        "predictions). Logistic Regression and Random Forest barely changed."),
    6: ("The models are compared on the same test set. The best classifier is chosen by <b>cross-validated "
        "AUC</b>, not by the test score, so the test set stays an honest final check. The confusion matrix shows "
        "the types of mistakes, and the Random Forest shows which features matter most. The best models are "
        "saved with joblib.",
        f"<b>Best classifier: Logistic Regression</b> (CV AUC {f3(tun['Logistic Regression']['CV_AUC'])}, test "
        f"accuracy {f3(tun['Logistic Regression']['Accuracy'])}, F1 {f3(tun['Logistic Regression']['F1'])}, "
        f"AUC {f3(tun['Logistic Regression']['ROC_AUC'])}). It correctly approved 53 and correctly rejected 21 "
        "applicants, but wrongly approved 19 rejected applicants and wrongly rejected 10 approved ones. "
        "<b>Best regressor: Linear Regression</b> (R2 0.760). Credit_Score is by far the most important "
        "feature for approval (0.39), followed by Loan_Amount and Annual_Income."),
}
FIGS = {
    3: [("01_regression_comparison", "Figure 1 - Left: test R2 of the four regression models. Right: predicted "
                                      "vs actual income for the chosen model; points on the dashed line are "
                                      "perfect predictions.", 0.95)],
    4: [("02_classification_metrics", "Figure 2 - Accuracy, F1 and ROC-AUC of the four default classifiers on "
                                       "the test set.", 0.92)],
    6: [("03_roc_curves", "Figure 3 - ROC curves of the tuned classifiers: the closer to the top-left corner, "
                          "the better. The dashed line is random guessing.", 0.62),
        ("04_confusion_matrices", "Figure 4 - Confusion matrices of the tuned classifiers on the test set "
                                  "(rows = actual, columns = predicted).", 0.97),
        ("05_feature_importance", "Figure 5 - The ten most important features in the Random Forest.", 0.78)],
}
EXTRA_TABLE = {3: reg_rows, 4: clf_rows, 5: tun_rows}
EXTRA_W = {3: [0.22, 0.11, 0.11, 0.14, 0.17, 0.25], 4: [0.25, 0.13, 0.13, 0.11, 0.1, 0.13, 0.15],
           5: [0.20, 0.34, 0.11, 0.13, 0.10, 0.12]}

st = []
# ---------- cover ----------
st += [Spacer(1, 35 * mm), P("Supervised Machine<br/>Learning Models", title_style),
       P("Week 2 Task Report - Regression and Classification with Scikit-learn", sub_style)]
st += table([
    ["Task", "Train at least two machine learning models and compare their performance using evaluation metrics"],
    ["Prepared by", AUTHOR], ["Date", DATE],
    ["Tools", "Python 3, Scikit-learn, Pandas, NumPy, Matplotlib"],
    ["Data", "Cleaned loan applicant dataset from Week 1 (515 rows, 15 columns)"],
    ["Models", "Linear Regression, Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors "
               "(5 algorithms, 8 trained models)"],
    ["Code", "src/train_models.py (full code and results in the GitHub repository)"],
], [0.2, 0.8])
st += [Spacer(1, 6), P("<b>Contents</b>", h2)]
st += bullets(["1. Objective and overview", "2. The five algorithms", "3. How models are evaluated",
               "4. Step-by-step documentation (Steps 1 to 6)", "5. Final model comparison",
               "6. Conclusion", "7. Limitations and next steps", "Appendix: repository and how to run"])
st.append(PageBreak())

# ---------- 1 objective ----------
st += [P("1. Objective and overview", h1),
       P("In supervised learning a model learns from examples where the correct answer is known, then predicts "
         "the answer for new data. This report covers the Week 2 objectives:")]
st += bullets([
    "Implement Linear Regression, Logistic Regression, Decision Tree, Random Forest and K-Nearest Neighbors "
    "with Scikit-learn.",
    "Learn model training, prediction and evaluation with performance metrics.",
    "<b>Deliverable:</b> train at least two models and compare their performance. Here, <b>eight models</b> "
    "are trained and compared.",
])
st.append(P("Two problems are solved with the cleaned dataset from Week 1: a <b>regression</b> problem "
            "(predict a loan applicant's annual income) and a <b>classification</b> problem (predict whether the "
            "loan is approved). Regression uses Linear Regression, KNN, Decision Tree and Random Forest. "
            "Classification uses Logistic Regression, KNN, Decision Tree and Random Forest."))

# ---------- 2 algorithms ----------
st += [P("2. The five algorithms", h1)]
st += table([
    ["Algorithm", "Used for", "How it works", "Strengths", "Weaknesses"],
    ["Linear Regression", "Regression", "Fits a weighted sum of the features that best matches the target number.",
     "Simple, fast, coefficients are easy to explain.", "Cannot capture curved or complex patterns."],
    ["Logistic Regression", "Classification",
     "Like linear regression, but an S-shaped function turns the result into a probability between 0 and 1.",
     "Simple, gives probabilities, hard to overfit.", "Only draws a straight boundary between classes."],
    ["Decision Tree", "Both", "Asks a chain of yes/no questions about the features to reach a prediction.",
     "Easy to read, handles non-linear patterns.", "Overfits easily; small data changes alter the tree."],
    ["Random Forest", "Both", "Trains many trees on random parts of the data and averages their predictions.",
     "Accurate, robust, shows feature importance.", "Slower and harder to explain."],
    ["K-Nearest Neighbors", "Both", "Predicts from the k most similar training examples.",
     "No training step, easy idea.", "Needs scaled features; sensitive to k and to irrelevant columns."],
], [0.15, 0.15, 0.27, 0.22, 0.21])

# ---------- 3 metrics ----------
st += [CondPageBreak(90 * mm), P("3. How models are evaluated", h1)]
st += table([
    ["Metric", "Problem", "Meaning"],
    ["MAE", "Regression", "Average size of the prediction error (lower is better)."],
    ["RMSE", "Regression", "Like MAE but punishes large errors more (lower is better)."],
    ["R2", "Regression", "Share of the variation in the target that the model explains: 1 is perfect, 0 is no "
                         "better than always predicting the average."],
    ["Accuracy", "Classification", "Share of all predictions that are correct."],
    ["Precision", "Classification", "Of the applicants predicted as approved, how many really were."],
    ["Recall", "Classification", "Of the applicants who really were approved, how many the model found."],
    ["F1 score", "Classification", "Balance of precision and recall in one number."],
    ["ROC-AUC", "Classification", "How well the model ranks approved above rejected applicants: 0.5 is guessing, "
                                  "1.0 is perfect."],
    ["Confusion matrix", "Classification", "Table of correct and wrong predictions for each class."],
    ["Cross-validation", "Both", "Trains and tests on 5 different splits of the training data and averages the "
                                 "score, so one lucky split does not decide."],
], [0.19, 0.16, 0.65])
st.append(CondPageBreak(110 * mm))

# ---------- 4 steps ----------
st.append(P("4. Step-by-step documentation", h1))
st.append(P("Each step shows what is done and why, the exact code that was run, its output, and the result. "
            "The code comes straight from <b>src/train_models.py</b>."))
for n in range(1, 7):
    why, result = STEP_TEXT[n]
    st += [CondPageBreak(75 * mm), P(f"Step {n}: {titles[n]}", h2), P(why),
           P("CODE", label), Preformatted(wrap(CODE[n]), code_style),
           P("OUTPUT", label), Preformatted(wrap(LOG[str(n)]), out_style)]
    if n in EXTRA_TABLE:
        cap = {3: "Regression results (sorted by cross-validated R2).",
               4: "Classification results with default settings (sorted by ROC-AUC).",
               5: "Tuned classifiers (sorted by cross-validated AUC)."}[n]
        st += [P(cap, label)] + table(EXTRA_TABLE[n], EXTRA_W[n])
    for name, cap, w in FIGS.get(n, []):
        st += image(name, cap, w)
    st += callout(f"<b>Result:</b> {result}")

# ---------- 5 final comparison ----------
st += [CondPageBreak(100 * mm), P("5. Final model comparison", h1),
       P("Regression (predict annual income)", h2)]
st += table(reg_rows, EXTRA_W[3])
st += [P("Classification (predict loan approval), tuned models", h2)]
st += table(tun_rows, EXTRA_W[5])
st += [P("<b>How to read this:</b> for both problems the simple linear models are as good as the more complex "
         "tree-based ones. When two models score almost the same, the simpler one is the better choice because "
         "it is faster, easier to explain and less likely to overfit.")]

# ---------- 6 conclusion ----------
st += [P("6. Conclusion", h1)]
st += bullets([
    f"<b>Regression:</b> Linear Regression predicts annual income with R2 {f3(reg[LR_]['R2'])} and an average "
    f"error of about Rs. {reg[LR_]['MAE_Rs']:,.0f}. Random Forest scored slightly higher on the test set "
    f"({f3(reg[RF]['R2'])}) but slightly lower in cross-validation, so the difference is not meaningful.",
    f"<b>Classification:</b> Logistic Regression is the best classifier by cross-validated AUC "
    f"({f3(tun['Logistic Regression']['CV_AUC'])}), with test accuracy "
    f"{f3(tun['Logistic Regression']['Accuracy'])} and AUC {f3(tun['Logistic Regression']['ROC_AUC'])}. "
    "Random Forest and KNN are very close.",
    "<b>Decision Tree</b> was the weakest model until it was tuned; a shallow tree (depth 2) generalised much "
    "better than a deep one.",
    "<b>Hyperparameter tuning</b> matters most for models that overfit or are sensitive to settings (Decision "
    "Tree, KNN).",
    "<b>Credit score</b> is the strongest signal for loan approval, which matches how lending works in practice.",
    "<b>Always compare several models</b> on the same split with several metrics, and choose with "
    "cross-validation rather than one test score.",
])

# ---------- 7 limitations ----------
st += [P("7. Limitations and next steps", h1)]
st += bullets([
    "The dataset is small (515 rows) and the test set has only 103 rows, so scores can move by several "
    "percentage points with a different split. Small gaps between models are not reliable.",
    "The data is synthetic and the approval label contains built-in randomness, so a score near 75% AUC is "
    "reasonable and perfect accuracy is not possible.",
    "In Week 1 the features were scaled on the whole dataset before splitting, which leaks a little "
    "information from the test rows. A cleaner approach is to put the scaler and the model in one Scikit-learn "
    "<b>Pipeline</b> so scaling is learned from the training data only.",
    "Next steps: use a Pipeline, try gradient boosting, handle class imbalance if it appears, and test on a real "
    "dataset.",
])

# ---------- appendix ----------
st += [P("Appendix: repository and how to run", h1)]
st += [Preformatted(
    "week2-supervised-ml-models/\n"
    "  README.md\n"
    "  requirements.txt\n"
    "  data/loan_applicants_clean.csv      cleaned data from Week 1\n"
    "  src/train_models.py                 trains, tunes and evaluates all models\n"
    "  src/build_report.py                 builds this PDF\n"
    "  results/                            metric tables (CSV), charts, log\n"
    "  models/                             saved best models (joblib)\n"
    "  report/                             this PDF", code_style),
    P("To run: <b>pip install -r requirements.txt</b>, then <b>python3 src/train_models.py</b> from the "
      "repository root. The script prints every result and writes the tables, charts and saved models.")]

doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                        bottomMargin=18 * mm, title="Week 2 - Supervised Machine Learning Models",
                        author=AUTHOR)
doc.build(st, onFirstPage=footer, onLaterPages=footer)
print("Built", PDF)
