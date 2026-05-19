import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# MILESTONE 3: TRAINING LOOP & METRICS

# 1. Load the cleaned dataset
cleaned_df = pd.read_csv('cleaned_dataset_numeric.csv')

# 2. Prepare Features (X) and Target (y) / Also use "depression" as the target variable
X = cleaned_df.drop('depression', axis=1)
y = cleaned_df['depression']

# Split data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Milestone 3: Training Loop Progress ---")

# Demonstrating how the model scales with more trees
tree_settings = [10, 50, 100, 150]
for n in tree_settings:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Trees: {n:3} | Validation Accuracy: {acc:.4f}")

# Final detailed evaluation for your Research Poster
print("\n--- Final Technical Metrics (150 Trees) ---")
final_model = RandomForestClassifier(n_estimators=150, random_state=42)
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

# This table provides the Precision and Recall
print(classification_report(y_test, final_pred))