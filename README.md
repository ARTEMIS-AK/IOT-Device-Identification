# IOT-Device-Identification

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load datasets
train = pd.read_csv('/content/drive/MyDrive/IOT ml/iot_device_train.csv')
test = pd.read_csv('/content/drive/MyDrive/IOT ml/iot_device_test.csv')

# Display first few rows of train and test datasets
print(train.head())
print(test.head())

# Display value counts of 'device_category' in the training set
print(train['device_category'].value_counts())

# Remove 'water_sensor' category from the training set
train = train[train['device_category'] != 'water_sensor']

# Concatenate train and test datasets and shuffle
frames = [train, test]
df = pd.concat(frames, ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

# Label encode categorical columns
for c in df.columns:
    if df[c].dtype == 'object':
        lbl = LabelEncoder()
        df[c] = lbl.fit_transform(df[c].values)

# Correlation analysis and dropping low correlation columns
all_corr = df.corr().abs()['device_category'].sort_values(ascending=False)
corr_drop = all_corr[all_corr < 0.0001]
to_drop = list(corr_drop.index)
df = df.drop(to_drop, axis=1)

# Splitting data into features and target variable
X = df.drop('device_category', axis=1)
y = df['device_category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a function to fit models and display their scores
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    model_scores = pd.DataFrame(model_scores, index=['Score']).transpose()
    model_scores = model_scores.sort_values('Score')
    return model_scores

# Define various models
models = {
    'LogisticRegression': LogisticRegression(max_iter=10000),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'XGBClassifier': XGBClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

# Evaluate baseline model performance
baseline_model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)
print(baseline_model_scores)

# Plot baseline model scores
plt.figure(figsize=(20,10))
sns.barplot(data=baseline_model_scores.sort_values('Score').T)
plt.title('Baseline Model Precision Score')
plt.xticks(rotation=90)
plt.show()

# Define ensemble voting classifier
ensemble_models = [
    ('randomforest', RandomForestClassifier()),
    ('XGBClassifier', XGBClassifier())
]
ABM = VotingClassifier(estimators=ensemble_models, voting='hard')

# Cross-validation for ensemble model
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(ABM, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(f'Ensemble model mean accuracy: {n_scores.mean()}')

# Fit the ensemble model and predict
ABM.fit(X_train, y_train)
y_preds = ABM.predict(X_test)

# Display classification report
print(classification_report(y_test, y_preds))
```

The script performs a comprehensive analysis and classification task using various machine learning models. Here's a description of the steps involved:

1. **Import Libraries:** Imports essential libraries and modules for data manipulation, visualization, and machine learning.

2. **Mount Google Drive:** Mounts Google Drive to access datasets stored there.

3. **Load Data:** Loads training and testing datasets from CSV files stored in Google Drive.

4. **Data Inspection:** Displays the first few rows of the training and testing datasets.

5. **Data Cleaning:** Removes rows where the 'device_category' is 'water_sensor' from the training dataset and then displays the value counts for the remaining categories.

6. **Data Concatenation:** Concatenates the training and testing datasets into a single dataframe and shuffles the data.

7. **Data Encoding:** Uses `LabelEncoder` to encode categorical features in the dataframe.

8. **Correlation Calculation:** Computes the correlation of all features with the 'device_category' target and drops features with negligible correlation.

9. **Data Splitting:** Splits the dataset into features (`X`) and target (`y`), and then further splits them into training and testing sets.

10. **Feature Scaling:** Applies `StandardScaler` to scale the feature values.

11. **Model Training and Evaluation:** Defines a function `fit_and_score` to train multiple models and evaluate their performance on the test set. The models used include Logistic Regression, K-Nearest Neighbors, Support Vector Classifier, Decision Tree, Random Forest, AdaBoost, XGBoost, and Gradient Boosting.

12. **Baseline Model Scores:** Trains the models and stores their accuracy scores, displaying these scores in a bar plot for comparison.

13. **Voting Classifier:** Sets up an ensemble model using a voting classifier with Random Forest and XGBoost, evaluates its performance using cross-validation, and prints the classification report with precision, recall, and f1-score metrics for each class.

The entire process covers data preprocessing, feature engineering, model training, evaluation, and ensemble learning to achieve robust classification performance.
