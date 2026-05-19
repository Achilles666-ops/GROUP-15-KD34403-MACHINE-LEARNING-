import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# LOAD DATA
df = pd.read_csv('cleaned_dataset_numeric.csv')

# FEATURES & SPLIT
X = df.drop('depression', axis=1)
y = df['depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MILESTONE 4: OPTIMIZATION
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
m = evaluate(rf_manual, "Manual Tuning")
print(classification_report(y_test, rf_manual.predict(X_test)))

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
print(classification_report(y_test, best.predict(X_test)))

# Feature Importance
importances = best.feature_importances_
feature_names = X.columns.tolist()
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
mean_imp = np.mean(importances)
print("\n--- Feature Importance ---")
for feat, score in feat_imp:
    print(f"  {'TOP' if score >= mean_imp else 'mod':3}  {feat:25} {score:.4f}")

# Summary
print(f"\n{'Stage':<20} {'Train':>7} {'Test':>7} {'CV':>7} {'Gap':>7}")
print("-" * 50)
for label, vals in [("Baseline", b), ("Manual Tuning", m), ("GridSearchCV", g)]:
    print(f"{label:<20} {vals[0]:>7.4f} {vals[1]:>7.4f} {vals[2]:>7.4f} {vals[3]:>7.4f}")
print(f"\nTop feature: {feat_imp[0][0]} ({feat_imp[0][1]:.4f})")
