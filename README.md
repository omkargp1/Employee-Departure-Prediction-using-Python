## 📊 Employee Departure Prediction using Python
This project addresses the challenge of predicting employee attrition using historical company data that lacks explicit departure labels. The solution simulates a real-world scenario where subject matter expert (SME) input is limited and must be used strategically to label data points.

### 🧠 Problem Statement

The goal is to develop a machine learning model to identify which employees are likely to leave the company. The dataset provided contains anonymized employee features without departure labels. Limited queries to a virtual SME (John) are allowed to label a subset of the data for training purposes.

### 🛠️ Key Features

* **Data Source**: Employee dataset from [UMBC Data Science repository](https://raw.githubusercontent.com/msaricaumbc/DS_data/master/ds602/final/employee_departure_dataset_X.csv)
* **Techniques Used**:

  * Data preprocessing and cleaning
  * Feature engineering
  * SMOTE for class imbalance
  * Ensemble models (Random Forest, Gradient Boosting)
  * Model evaluation using F1-score and ROC AUC
  * Model serialization with `joblib`

### 📁 Structure

This repository includes:

* Jupyter notebook with the full analysis and modeling pipeline
* Final trained model for inference
