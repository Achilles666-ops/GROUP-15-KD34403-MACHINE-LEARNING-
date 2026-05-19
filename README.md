# GROUP-15-MACHINE-LEARNING
Machine learning pipeline
🧠Prediction of student range of CGPA Using Machine Learning

Predicting student CGPA range based on their mental health parameters using supervised machine learning techniques.


📌 Project Overview
This project aims to predict student CGPA range based on mental health-related parameters using machine learning techniques. The dataset used is mental_health.csv, which contains various attributes related to an individual's mental condition, lifestyle, and emotional state.
Objectives

📊 Analyze mental health data
🧹 Preprocess and clean the dataset
🤖 Train a machine learning model
⚙️ Optimize model performance
📈 Evaluate prediction accuracy

-How to RUN the project in Google Colab-

STEP 1:
Download the dataset "mental_health.csv"

Make sure the dataset contains all required columns like:
depression, anxiety, isolation, academic_pressure, cgpa etc.

## 🚀 How to Run the Project in Google Colab

Follow these steps to run each milestone of the project in Google Colab.

---
Step 2: Open Google Colab

Go to [https://colab.research.google.com](https://colab.research.google.com) and sign in with your Google account.

---
Step 3: Clone the Repository

In a new Colab notebook, run the following cell to clone this repository:

```python
!git clone https://github.com/Achilles666-ops/GROUP-15-KD34403-MACHINE-LEARNING-.git
%cd GROUP-15-KD34403-MACHINE-LEARNING-
```

---

Step 4: Upload the Dataset (if needed)

The dataset `mental_health.csv` is already included in the repository. After cloning, it will be available in your working directory. You can verify it with:

```python
import os
print(os.listdir())
```

If it is missing, manually upload `mental_health.csv` using:

```python
from google.colab import files
uploaded = files.upload()
```

---

Step 5: Install Required Libraries

Most libraries are pre-installed in Colab. If any are missing, install them with:

```python
!pip install pandas scikit-learn matplotlib seaborn
```

---

Step 6: Run the Milestone Files

Run each milestone script in order using the `exec()` function or by copying the code into Colab cells.

**Milestone 1 – Data Preprocessing:**
```python
exec(open("Milestone 1 Code.py").read())
```

**Milestone 3 – Model Training:**
```python
exec(open("Milestone 3 code.py").read())
```

**Milestone 4 – Model Optimization:**
```python
exec(open("Milestone 4 Code.py").read())
```

**Milestone 5 – Model Evaluation:**
```python
exec(open("Milestone 5 Code.py").read())
```


Step 7: View Results

After running each script, outputs such as accuracy scores, confusion matrices, and plots will be displayed directly in the Colab notebook.
