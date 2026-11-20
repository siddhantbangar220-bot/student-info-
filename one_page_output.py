"""
One-page output generator for the ML mini project.
Usage:  python one_page_output.py --csv responses.csv
Creates: Output_OnePage.pdf  (A4, all key results on a single page)
Same column names as student_performance_ml.py.
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

NUM = ["study_hours","sleep_hours","phone_hours","attendance_pct","backlogs","stress_level"]
CAT = ["extracurricular","study_mode","online_courses"]
TARGET = "cgpa"; ORDER = ["Low","Medium","High"]
band = lambda c: "Low" if c < 6.5 else ("Medium" if c < 8 else "High")

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True)
ap.add_argument("--out", default="Output_OnePage.pdf"); a = ap.parse_args()

df = pd.read_csv(a.csv).drop_duplicates().dropna(subset=[TARGET])
df = df[df.sleep_hours.between(2,14) & df.study_hours.between(0,16) & df.attendance_pct.between(0,100) & df[TARGET].between(0,10)]
df["cgpa_band"] = df[TARGET].apply(band)
X, y = df[NUM+CAT], df["cgpa_band"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pre = ColumnTransformer([
 ("num", Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]), NUM),
 ("cat", Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]), CAT)])
models = {"Logistic Regression": LogisticRegression(max_iter=1000), "KNN": KNeighborsClassifier(n_neighbors=7),
          "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42)}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows, fit = [], {}
for n, m in models.items():
    p = Pipeline([("pre",pre),("m",m)])
    cva = cross_val_score(p, Xtr, ytr, cv=cv).mean(); p.fit(Xtr, ytr); pr = p.predict(Xte)
    rows.append([n, f"{cva:.3f}", f"{accuracy_score(yte,pr):.3f}", f"{f1_score(yte,pr,average='macro'):.3f}"])
    fit[n] = (p, pr)
best = max(rows, key=lambda r: float(r[3]))[0]; bp, bpred = fit[best]

sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(8.27, 11.69))
gs = GridSpec(4, 2, figure=fig, height_ratios=[0.5,1.3,1.3,1.3], hspace=0.55, wspace=0.35, left=0.08, right=0.95, top=0.96, bottom=0.05)
t = fig.add_subplot(gs[0,:]); t.axis("off")
t.text(0, 0.8, "Student Performance Predictor - Output Summary", fontsize=15, fontweight="bold")
t.text(0, 0.35, f"Samples used: {len(df)}   |   Train/Test: {len(Xtr)}/{len(Xte)}   |   Best model: {best}   |   Test accuracy: {accuracy_score(yte,bpred):.3f}", fontsize=9)

ax = fig.add_subplot(gs[1,0]); sns.countplot(x="cgpa_band", data=df, order=ORDER, ax=ax); ax.set_title("Class distribution", fontsize=10); ax.set_xlabel("")
ax = fig.add_subplot(gs[1,1]); sns.heatmap(df[NUM+[TARGET]].corr(), annot=True, fmt=".1f", cmap="coolwarm", cbar=False, annot_kws={"size":6}, ax=ax)
ax.set_title("Correlation heatmap", fontsize=10); ax.tick_params(labelsize=6)

ax = fig.add_subplot(gs[2,0]); ax.axis("off"); ax.set_title("Model comparison", fontsize=10)
tb = ax.table(cellText=rows, colLabels=["Model","CV Acc","Test Acc","Test F1"], loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.auto_set_column_width(list(range(5))); tb.scale(1, 1.6)
for (r,c),cell in tb.get_celld().items():
    if r == 0: cell.set_facecolor("#1F3A5F"); cell.set_text_props(color="white", fontweight="bold")
ax = fig.add_subplot(gs[2,1])
cm = confusion_matrix(yte, bpred, labels=ORDER)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=ORDER, yticklabels=ORDER, ax=ax)
ax.set_title(f"Confusion matrix - {best}", fontsize=10); ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)

rf = fit["Random Forest"][0]
imp = pd.Series(rf.named_steps["m"].feature_importances_, index=rf.named_steps["pre"].get_feature_names_out()).sort_values().tail(8)
ax = fig.add_subplot(gs[3,0]); imp.plot(kind="barh", ax=ax); ax.set_title("Top features (Random Forest)", fontsize=10); ax.tick_params(labelsize=7)
ax = fig.add_subplot(gs[3,1]); ax.axis("off"); ax.set_title(f"Per-class report - {best}", fontsize=10)
p_, r_, f_, s_ = precision_recall_fscore_support(yte, bpred, labels=ORDER, zero_division=0)
rep = [[o, f"{p_[i]:.2f}", f"{r_[i]:.2f}", f"{f_[i]:.2f}", int(s_[i])] for i,o in enumerate(ORDER)]
tb = ax.table(cellText=rep, colLabels=["Class","Precision","Recall","F1","Support"], loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.auto_set_column_width(list(range(5))); tb.scale(1, 1.6)
for (r,c),cell in tb.get_celld().items():
    if r == 0: cell.set_facecolor("#1F3A5F"); cell.set_text_props(color="white", fontweight="bold")
fig.savefig(a.out, format="pdf"); print("Saved", a.out)
