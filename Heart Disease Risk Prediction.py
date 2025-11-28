#Aad datasets from the UCI Machine Learning Repository more easily
#ucimlrepo is a Python package that provides functions to fetch datasets directly from the UCI ML Repository.
#This can save you the hassle of manually downloading CSV files, uploading them, and reading them in.
#You can programmatically load the data.
################################Install it######################################################################
# Step 0: If you want to run this directly inside a Jupyter notebook cell use the code below with percentage mark, otherwise you 
#get this syntax error pip install ucimlrepo ^ SyntaxError: invalid syntax, because pip install ... is a shell command, 
#not Python code.

%pip install ucimlrepo

#or this !pip install ucimlrepo   The ! tells Jupyter to run it as a shell command, not Python.

#########################Load the Heart Disease dataset#########################################################
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ucimlrepo import fetch_ucirepo

# Step 2: Load Heart Disease Dataset
heart = fetch_ucirepo(id=45)  # UCI Heart Disease dataset
X = heart.data.features
y = heart.data.targets

# Convert target to Series if needed
if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
    y = y.iloc[:, 0]

# Step 3: Convert to Binary Classification
# 0 = no heart disease, 1 = heart disease (any level > 0)
y_bin = (y > 0).astype(int)

# Step 4: Handle missing values
imputer = SimpleImputer(strategy='mean')  # replace NaN with column mean
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Step 5: Explore Data
print(X_imputed.info())
print(X_imputed.describe())
print(y_bin.value_counts())

sns.countplot(x=y_bin)
plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
plt.show()

# Step 6: Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_bin, test_size=0.2, random_state=42
)

# Step 7: Scale Features (needed for Logistic Regression & Gradient Boosting)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train and Evaluate Models
def train_evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    acc = accuracy_score(y_te, y_pred)
    roc = roc_auc_score(y_te, y_pred)
    print(f"{model_name} Accuracy: {acc:.2f}, ROC-AUC: {roc:.2f}")
    print(classification_report(y_te, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    
    return y_pred

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
y_pred_dt = train_evaluate_model(dt, X_train, X_test, y_train, y_test, "Decision Tree")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred_rf = train_evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest")

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
y_pred_lr = train_evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
y_pred_gb = train_evaluate_model(gb, X_train, X_test, y_train, y_test, "Gradient Boosting")

# Step 9: Feature Importance for Random Forest
importances = pd.Series(rf.feature_importances_, index=X_imputed.columns)
importances.sort_values().plot(kind='barh', title="Random Forest Feature Importance")
plt.show()
