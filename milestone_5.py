import pandas as pd
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

# =======================================================
# MILESTONE 1: DATA PREPROCESSING
# =======================================================

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # 1. Trim whitespace dari nama kolum
    df.columns = df.columns.str.strip()

    # 2. Trim whitespace dalam data
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # 3. Filter invalid CGPA
    df = df[df['cgpa'] != "0.0-0.0"]

    # 4. Buang baris kosong
    df = df.dropna()

    # 5. Encoding Mappings
    gender_map = {"Male": 0, "Female": 1}
    year_map = {"1st year": 1, "2nd year": 2, "3rd year": 3, "4th year": 4}
    cgpa_map = {
        "1.5-2.0": 1.75, "2.0-2.5": 2.25, "2.5-3.0": 2.75,
        "3.0-3.5": 3.25, "3.5-4.0": 3.75
    }

    if df['gender'].dtype == 'object':
        df['gender'] = df['gender'].map(gender_map)
    if df['academic_year'].dtype == 'object':
        df['academic_year'] = df['academic_year'].map(year_map)
    if df['cgpa'].dtype == 'object':
        df['cgpa'] = df['cgpa'].map(cgpa_map)

    return df.dropna()

# Load data
df = preprocess_data('cleaned_dataset_numeric.csv')
print(f"Preprocessing complete! Rows: {len(df)}")

# ── FEATURES & SPLIT ─────────────────────────────────────
mental_features = [
    'depression', 'anxiety', 'isolation', 'future_insecurity',
    'academic_pressure', 'academic_workload', 'financial_concerns',
    'social_relationships', 'study_satisfaction'
]
X = df[mental_features]
y = df['cgpa'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =======================================================
# MILESTONE 3: TRAINING LOOP
# =======================================================
print("\n--- M3: Training Loop (Predicting CGPA) ---")
for n in [10, 50, 100, 150]:
    m = RandomForestClassifier(n_estimators=n, random_state=42)
    m.fit(X_train, y_train)
    print(f"Trees: {n:3} | Accuracy: {accuracy_score(y_test, m.predict(X_test)):.4f}")

final_model = RandomForestClassifier(n_estimators=150, random_state=42)
final_model.fit(X_train, y_train)
print(classification_report(y_test, final_model.predict(X_test), zero_division=0))

# =======================================================
# MILESTONE 4: MODEL OPTIMIZATION
# =======================================================
def evaluate(model, label):
    t0 = time.time()
    model.fit(X_train, y_train)
    t = time.time() - t0
    tr = accuracy_score(y_train, model.predict(X_train))
    te = accuracy_score(y_test,  model.predict(X_test))
    cv = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    print(f"  [{label}] Train={tr:.4f}  Test={te:.4f}  CV={cv:.4f}  Gap={tr-te:.4f}  Time={t:.2f}s")
    return tr, te, cv, tr - te, t

print("\n=== MILESTONE 4: OPTIMIZATION ===")
b = evaluate(RandomForestClassifier(n_estimators=150, random_state=42), "Baseline")

rf_manual = RandomForestClassifier(
    n_estimators=150, max_depth=4, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=42
)
m_result = evaluate(rf_manual, "Manual Tuning")
print(classification_report(y_test, rf_manual.predict(X_test), zero_division=0))

# GridSearchCV
param_grid = {
    'n_estimators':      [50, 100, 150, 200],
    'max_depth':         [3, 4, 5, None],
    'min_samples_split': [3, 5, 8],
    'min_samples_leaf':  [1, 2, 3],
    'max_features':      ['sqrt', 'log2'],
}
t0 = time.time()
gs = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                  param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)
best = gs.best_estimator_
g = evaluate(best, "GridSearchCV")
print(f"  Best params: {gs.best_params_}")
print(classification_report(y_test, best.predict(X_test), zero_division=0))

# Feature Importance (text)
importances = best.feature_importances_
feat_imp = sorted(zip(mental_features, importances), key=lambda x: x[1], reverse=True)
mean_imp = np.mean(importances)
print("\n--- Feature Importance ---")
for feat, score in feat_imp:
    print(f"  {'TOP' if score >= mean_imp else 'mod':3}  {feat:25} {score:.4f}")

# Milestone 4 Summary Table
print(f"\n{'Stage':<20} {'Train':>7} {'Test':>7} {'CV':>7} {'Gap':>7}")
print("-" * 50)
for label, vals in [("Baseline", b), ("Manual Tuning", m_result), ("GridSearchCV", g)]:
    print(f"{label:<20} {vals[0]:>7.4f} {vals[1]:>7.4f} {vals[2]:>7.4f} {vals[3]:>7.4f}")
print(f"\nTop feature: {feat_imp[0][0]} ({feat_imp[0][1]:.4f})")

# =======================================================
# MILESTONE 5: FINAL EVALUATION
# =======================================================
print("\n" + "="*55)
print("      MILESTONE 5: FINAL MODEL EVALUATION")
print("="*55)

# Use the BEST model from GridSearchCV
# Evaluate on TEST SET only (data never seen during training)
y_pred_final = best.predict(X_test)
y_pred_train = best.predict(X_train)

test_acc  = accuracy_score(y_test, y_pred_final)
train_acc = accuracy_score(y_train, y_pred_train)
cv_final  = cross_val_score(best, X, y, cv=5, scoring='accuracy')

print(f"\n  Final Train Accuracy : {train_acc:.4f}")
print(f"  Final Test Accuracy  : {test_acc:.4f}")
print(f"  CV Mean Accuracy     : {cv_final.mean():.4f} (+/- {cv_final.std():.4f})")
print(f"  Overfitting Gap      : {train_acc - test_acc:.4f}")
print(f"\n  Best Parameters Used : {gs.best_params_}")

print("\n--- Final Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred_final, zero_division=0))

# ── CHART 1: Confusion Matrix ─────────────────────────────
print("\nGenerating charts...")
cm = confusion_matrix(y_test, y_pred_final, labels=best.classes_)
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(confusion_matrix=cm,
    display_labels=best.classes_).plot(ax=ax, cmap='Blues', colorbar=True)
ax.set_title('Milestone 5 – Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('m5_confusion_matrix.png', dpi=150)
plt.show()
print("Saved: m5_confusion_matrix.png")

# ── CHART 2: Feature Importance ───────────────────────────
feat_df = pd.DataFrame({
    'Feature': mental_features,
    'Importance': best.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#e74c3c' if v >= feat_df['Importance'].mean()
          else '#3498db' for v in feat_df['Importance']]
ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors)
ax.axvline(feat_df['Importance'].mean(), color='gray',
           linestyle='--', label='Mean Importance')
ax.set_title('Milestone 5 – Feature Importance (Best Model)', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.legend()
plt.tight_layout()
plt.savefig('m5_feature_importance.png', dpi=150)
plt.show()
print("Saved: m5_feature_importance.png")

# ── CHART 3: Cross Validation Scores ─────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar([f'Fold {i+1}' for i in range(5)], cv_final,
              color='#2ecc71', edgecolor='black', width=0.5)
ax.axhline(cv_final.mean(), color='red', linestyle='--',
           label=f'Mean = {cv_final.mean():.4f}')
ax.set_ylim(0, 1)
ax.set_title('Milestone 5 – Cross-Validation Accuracy', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy')
ax.legend()
for bar, val in zip(bars, cv_final):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('m5_cross_validation.png', dpi=150)
plt.show()
print("Saved: m5_cross_validation.png")

# ── CHART 4: Train vs Test vs CV Accuracy Summary ────────
fig, ax = plt.subplots(figsize=(6, 4))
labels = ['Train Accuracy', 'Test Accuracy', 'CV Mean']
values = [train_acc, test_acc, cv_final.mean()]
bars = ax.bar(labels, values,
              color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
ax.set_ylim(0, 1)
ax.set_title('Milestone 5 – Final Accuracy Summary', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01, f'{val:.4f}',
            ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig('m5_accuracy_summary.png', dpi=150)
plt.show()
print("Saved: m5_accuracy_summary.png")

print("\n✅ ALL MILESTONES COMPLETE!")
print("📁 4 chart images saved in your folder:")
print("   - m5_confusion_matrix.png")
print("   - m5_feature_importance.png")
print("   - m5_cross_validation.png")
print("   - m5_accuracy_summary.png")