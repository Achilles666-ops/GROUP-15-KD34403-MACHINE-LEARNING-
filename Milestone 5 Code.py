import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

#Load and Preprocess Data 
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df[df['cgpa'] != "0.0-0.0"]
    df = df.dropna()
    if df['gender'].dtype == 'object':
        df['gender'] = df['gender'].map({"Male": 0, "Female": 1})
    if df['academic_year'].dtype == 'object':
        df['academic_year'] = df['academic_year'].map({"1st year": 1, "2nd year": 2, "3rd year": 3, "4th year": 4})
    if df['cgpa'].dtype == 'object':
        df['cgpa'] = df['cgpa'].map({"1.5-2.0": 1.75, "2.0-2.5": 2.25, "2.5-3.0": 2.75, "3.0-3.5": 3.25, "3.5-4.0": 3.75})
    return df.dropna()

df = preprocess_data('cleaned_dataset_numeric.csv')

#Features and Split
mental_features = [
    'depression', 'anxiety', 'isolation', 'future_insecurity',
    'academic_pressure', 'academic_workload', 'financial_concerns',
    'social_relationships', 'study_satisfaction'
]
X = df[mental_features]
y = df['cgpa'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Load Best Model from Milestone 4 

param_grid = {
    'n_estimators':      [50, 100, 150, 200],
    'max_depth':         [3, 4, 5, None],
    'min_samples_split': [3, 5, 8],
    'min_samples_leaf':  [1, 2, 3],
    'max_features':      ['sqrt', 'log2'],
}
gs = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
gs.fit(X_train, y_train)
best = gs.best_estimator_

# =======================================================
# MILESTONE 5: EVALUATE ON TEST SET ONLY
# =======================================================


y_pred_final = best.predict(X_test)
y_pred_train = best.predict(X_train)


test_acc  = accuracy_score(y_test, y_pred_final)
train_acc = accuracy_score(y_train, y_pred_train)
cv_final  = cross_val_score(best, X, y, cv=5, scoring='accuracy')


print("=" * 55)
print("      MILESTONE 5: FINAL MODEL EVALUATION")
print("=" * 55)
print(f"\n  Final Train Accuracy : {train_acc:.4f}")
print(f"  Final Test Accuracy  : {test_acc:.4f}")
print(f"  CV Mean Accuracy     : {cv_final.mean():.4f} (+/- {cv_final.std():.4f})")
print(f"  Overfitting Gap      : {train_acc - test_acc:.4f}")
print(f"\n  Best Parameters Used : {gs.best_params_}")
print("\n--- Final Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred_final, zero_division=0))

# CHART 1
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

#  CHART 2
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

#CHART 3
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

#CHART 4
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

print("\n Milestone 5 Complete!")
print(" 4 chart images saved:")
print("   - m5_confusion_matrix.png")
print("   - m5_feature_importance.png")
print("   - m5_cross_validation.png")
print("   - m5_accuracy_summary.png")
