# IOT-Device-Identification

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
