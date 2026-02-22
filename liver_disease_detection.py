import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             ConfusionMatrixDisplay)
# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
# Load dataset [cite: 569]
df = pd.read_csv('DatasetProject.csv')
# Handling missing values [cite: 576, 577]
# Imputing missing values in Albumin_Globulin_Ratio with the mean
df['Albumin_Globulin_ Ratio'].fillna(df['Albumin_Globulin_ Ratio'].mean(), inplace=True)
# Encoding categorical variables [cite: 583]
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender']) # Transforms 'Female'/'Male' to numerical [cite: 585]
# Defining Features and Target [cite: 593, 598]
# Removing Patient_ID as it doesn't contribute to prediction [cite: 594, 608]
X = df.drop(['Patient_ID', 'Liver_Diseases'], axis=1) 
y = df['Liver_Diseases']
# Scaling Features [cite: 589]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ==========================================
# 2. Training and Testing Split
# ==========================================
# Split: 70% Training, 30% Testing [cite: 609, 611]
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# ==========================================
# 3. Model Training & Evaluation
# ==========================================
# Dictionary to store all classifiers [cite: 1152]
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}
results = []
print("Classification Results:\n" + "="*30)
for name, clf in classifiers.items():
    # Train the model [cite: 698]
    clf.fit(X_train, y_train)
    # Make predictions [cite: 745, 809, 876]
    y_pred = clf.predict(X_test)  
    # Calculate Metrics [cite: 702, 708]
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_pred) * 100  
    results.append({
        "Model": name,
        "Accuracy (%)": round(acc, 2),
        "Precision (%)": round(prec, 2),
        "Recall (%)": round(rec, 2),
        "F1-Score (%)": round(f1, 2),
        "ROC-AUC (%)": round(auc, 2)
    })
        print(f"{name} Accuracy: {acc:.2f}%")
# ==========================================
# 4. Comparative Analysis Table [cite: 1120]
# ==========================================
df_results = pd.DataFrame(results)
print("\nSummary Table:")
print(df_results)
# ==========================================
# 5. Visualizing Best Model [cite: 1205]
# ==========================================
best_model_name = df_results.loc[df_results['Accuracy (%)'].idxmax()]['Model']
print(f"\nBest Model identified by accuracy: {best_model_name}") [cite: 1210]
# Plot Actual vs Predicted for all techniques [cite: 1159, 1171]
plt.figure(figsize=(14, 8))
for name, clf in classifiers.items():
    y_pred = clf.predict(X_test)
    plt.scatter(range(len(y_pred)), y_pred, label=f'{name} Predicted', marker='+')
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', marker='o', alpha=0.5)
plt.title('Actual vs Predicted Values for All Techniques') [cite: 1171]
plt.xlabel('Index') [cite: 1172]
plt.ylabel('Liver Disease (0: No, 1: Yes)') [cite: 1173]
plt.legend(loc='best') [cite: 1174]
plt.show()
