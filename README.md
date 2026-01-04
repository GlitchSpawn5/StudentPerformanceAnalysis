STUDENT PERFORMANCE ANALYTICS – PROJECT DOCUMENTATION
====================================================

AUTHOR: Hendric Nazarius
PROJECT TYPE: Machine Learning (Regression + Classification)

----------------------------------------------------
1. PROJECT OVERVIEW
----------------------------------------------------
This project focuses on analyzing and predicting student academic performance
using supervised machine learning techniques. Regression and classification
models are used to understand how academic factors influence student outcomes.

Time-series analysis and attendance-based modeling were intentionally excluded
due to the absence of genuine temporal data in the dataset.

----------------------------------------------------
2. OBJECTIVES
----------------------------------------------------
• Predict students’ final scores using Regression
• Classify students as Pass / Fail using Classification
• Perform basic exploratory data analysis (EDA)

----------------------------------------------------
3. MACHINE LEARNING TASKS
----------------------------------------------------

3.1 REGRESSION
Algorithm      : Linear Regression
Target Variable: Final Score
Purpose        : Predict numerical student performance

3.2 CLASSIFICATION
Algorithm      : Logistic Regression
Target Variable: Pass / Fail
Purpose        : Identify students at academic risk

----------------------------------------------------
4. DATASET
----------------------------------------------------
Source: Kaggle – Student Performance Dataset

The dataset contains academic and demographic attributes such as:
• Study hours
• Midterm scores
• Final scores
• Other student-related features

NOTE:
Time-series analysis was excluded due to the absence of temporal data.

----------------------------------------------------
5. PROJECT STRUCTURE
----------------------------------------------------

student-performance-analytics/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── regression.py
│   └── classification.py
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── models/
│   ├── regression_model.pkl
│   └── classification_model.pkl
│
├── README.md
├── requirements.txt
└── .gitignore

----------------------------------------------------
6. FILE RESPONSIBILITIES
----------------------------------------------------

Folders:
• data/        → stores raw and cleaned datasets
• src/         → contains ML source code
• notebooks/   → exploratory data analysis
• models/      → saved trained models

Important Notes:
• .pkl files are auto-generated when scripts are run
• notebooks can be simple (EDA only)
• README.md explains the project
• requirements.txt lists dependencies

----------------------------------------------------
7. INSTALLATION & SETUP
----------------------------------------------------

Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Run regression model
python src/regression.py

Step 3: Run classification model
python src/classification.py

----------------------------------------------------
8. MODEL EVALUATION
----------------------------------------------------

Regression Metrics:
• RMSE
• R² Score

Classification Metrics:
• Accuracy
• Confusion Matrix
• Precision
• Recall
• F1-score

Trained models are saved in the models/ directory.

----------------------------------------------------
9. EXPLORATORY DATA ANALYSIS
----------------------------------------------------
EDA is performed in:
notebooks/01_eda.ipynb

Typical EDA includes:
• Dataset overview
• Summary statistics
• Feature distributions
• Correlation analysis

----------------------------------------------------
10. TOOLS & TECHNOLOGIES
----------------------------------------------------
• Python
• Pandas
• NumPy
• Scikit-learn
• Matplotlib
• Seaborn
• Jupyter Notebook

----------------------------------------------------
11. CONCLUSION
----------------------------------------------------
This project demonstrates the application of supervised machine learning
techniques to analyze and predict student academic performance. The models
provide insights into key factors influencing student success and form a solid
foundation for further extensions.

====================================================
END OF DOCUMENT
====================================================
