# Heart-Disease-Risk-Prediction-Python-Jupyter-Notebook
This Jupyter Notebook demonstrates how to predict the risk of heart disease using the UCI Heart Disease dataset.It implements multiple machine learning algorithms, including:
Decision Tree
Random Forest
Logistic Regression
Gradient Boosting Classifier

The notebook is fully self-contained and does not require manually downloading the dataset from UCI. Instead, it uses the Python package ucimlrepo to fetch the dataset programmatically.
Key Features and Fixes
1- Automatic dataset download via ucimlrepo
%pip install ucimlrepo installs the package.
fetch_ucirepo(id=45) retrieves the Heart Disease dataset directly.
No manual download or file path management is required.

2- Handling missing values
The UCI dataset contains some missing values (NaN).
It uses SimpleImputer(strategy='mean') to replace missing values with the mean of the respective column.
This ensures that models like Decision Tree and Random Forest can run without errors.

3- Binary classification of target variable
Original dataset has multiple target values indicating disease severity (0–4).
It converts all target > 0 to 1 (heart disease present), and 0 remains 0 (no disease).
This resolves errors with roc_auc_score and simplifies risk prediction.

4- Data preprocessing
Features are optionally scaled using StandardScaler for algorithms that benefit from scaling (Logistic Regression, Gradient Boosting).
Train/test split is performed using train_test_split with 80/20 ratio.

5- Model training and evaluation
Each model is trained on the training set and evaluated on the test set.
Metrics included: Accuracy, ROC-AUC, Precision, Recall, F1-score.
Confusion matrices are plotted for visual evaluation.

6- Feature importance visualization
For Random Forest, feature importance is calculated and plotted using rf.feature_importances_. 
Helps identify which features contribute most to predicting heart disease risk.

How to Run

1-Open the notebook in Jupyter or VS Code.
2-Run the first cell to install ucimlrepo.
3-Run all cells sequentially, the notebook will:
Fetch the dataset automatically
Handle missing values
Train multiple models
Evaluate performance metrics and confusion matrices
Visualize feature importance
No additional setup is needed.
