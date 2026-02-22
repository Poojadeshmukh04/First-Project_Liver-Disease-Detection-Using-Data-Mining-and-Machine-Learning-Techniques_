# First-Project_Liver-Disease-Detection-Using-Data-Mining-and-Machine-Learning-Techniques_
This project focuses on the early detection of liver diseases using clinical and biochemical parameters. By analyzing data such as bilirubin concentrations and liver enzyme measurements (ALT, AST), machine learning models were developed to provide a non-invasive and accurate diagnostic tool.
# Dataset
The research utilizes the Indian Liver Patient Dataset (ILPD) sourced from the UCI Machine Learning Repository.
Total Records: 579 
Key Features: Age, Gender, Total Bilirubin, Alkaline Phosphatase, Albumin, and Liver Enzymes.
# Methodology
I implemented and compared six different machine learning algorithms to determine the most effective diagnostic approach:
Logistic Regression
Decision Tree Classifier
Random Forest
K-Nearest Neighbors (KNN)
Gradient Boosting
Naive Bayes
# Results
The Random Forest model emerged as the most accurate for liver disease detection :
# Model Performance Comparison
I evaluated six models to find the most effective diagnostic tool.
While **Random Forest** achieved the highest overall accuracy, 
**Naive Bayes** showed the best ability to identify actual disease cases (Recall).
<br>
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| **Random Forest** | **72.57** | **48.48** | 34.04 | 40.00 | 60.38 |
<br>
| Logistic Regression | 72.00 | 45.00 | 19.15 | 26.87 | 55.28 |
<br>
| Gradient Boosting | 70.86 | 44.74 | 36.17 | 40.00 | 59.88 |
<br>
| Decision Tree | 69.14 | 43.40 | 48.94 | 46.00 | 62.75 |
<br>
| K-Nearest Neighbors | 65.71 | 37.25 | 40.43 | 38.78 | 57.71 |
<br>
| Naive Bayes | 57.14 | 38.33 | **97.87** | 55.09 | 70.03 |
# Tools & Technologies
Language: Python 
Libraries: Pandas, Scikit-Learn, Matplotlib, Seaborn 
IDE: Spyder
