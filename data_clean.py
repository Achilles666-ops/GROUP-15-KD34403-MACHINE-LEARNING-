import pandas as pd

input_file = "mental_health.csv"
output_file = "cleaned_dataset_numeric.csv"

# --- Load the dataset ---
df = pd.read_csv(input_file)

# --- Trim whitespace from all string columns ---
df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# --- Remove rows with invalid CGPA ---
df = df[df["cgpa"] != "0.0-0.0"]

# --- Remove rows with any missing/empty fields ---
df = df.dropna()
df = df[~df.apply(lambda row: row.astype(str).str.strip().eq("").any(), axis=1)]

removed_rows = len(pd.read_csv(input_file)) - len(df)

# --- Encode categorical columns ---
gender_map = {"Male": 0, "Female": 1}
year_map = {"1st year": 1, "2nd year": 2, "3rd year": 3, "4th year": 4}
cgpa_map = {
    "0.0-0.0": 0.0,
    "1.5-2.0": 1.75,
    "2.0-2.5": 2.25,
    "2.5-3.0": 2.75,
    "3.0-3.5": 3.25,
    "3.5-4.0": 3.75,
}
residential_map = {"Off-Campus": 0, "On-Campus": 1}
yes_no_map = {"No": 0, "Yes": 1}
sports_map = {"No Sports": 0, "1-3 times": 1, "4-6 times": 2, "7+ times": 3}
sleep_map = {"2-4 hrs": 3, "4-6 hrs": 5, "7-8 hrs": 7}

df["gender"] = df["gender"].map(gender_map)
df["academic_year"] = df["academic_year"].map(year_map)
df["cgpa"] = df["cgpa"].map(cgpa_map)
df["residential_status"] = df["residential_status"].map(residential_map)
df["campus_discrimination"] = df["campus_discrimination"].map(yes_no_map)
df["sports_engagement"] = df["sports_engagement"].map(sports_map).fillna(0).astype(int)
df["average_sleep"] = df["average_sleep"].map(sleep_map)

# --- Save cleaned dataset ---
df.to_csv(output_file, index=False)

print("Preprocessing complete!")
print(f"Removed rows: {removed_rows}")
print(f"Remaining rows: {len(df)}")
print("\nSample cleaned numeric data:")
print(df.head(5).to_string())