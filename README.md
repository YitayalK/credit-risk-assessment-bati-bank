---

```markdown
# Credit Risk Analysis using Machine Learning

## Overview

This project focuses on building a credit scoring model to assess credit risk for a new buy-now-pay-later service at Bati Bank. The aim is to classify users into high risk (bad) and low risk (good) categories, develop a probability-based risk scoring system, and predict the optimal loan amount and duration. The solution involves data exploration, feature engineering, model training with hyperparameter tuning, and deploying the trained model via a REST API.

## Business Need

Bati Bank is partnering with an eCommerce platform to offer a credit-based payment solution. With historical transaction data available, this project helps:
- **Define a Proxy Variable:** Categorize users based on risk.
- **Feature Engineering:** Create predictive features using transaction history, date-time details, and domain-specific aggregations.
- **Model Development:** Use ML models (e.g., Logistic Regression, Random Forest, GBM) to estimate risk probabilities and assign credit scores.
- **API Deployment:** Serve the model predictions in real time using a REST API.

## Repository Structure

```plaintext
Credit-Risk-Analysis/
├── README.md
├── data/
│   ├── train.csv            # Training dataset with transaction details
│   └── test.csv             # Test dataset with similar structure
├── notebooks/
│   ├── EDA.ipynb            # Notebook for Exploratory Data Analysis
│   └── Feature_Engineering_Modeling.ipynb  # Notebook for feature engineering, model training, and evaluation
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data loading and cleaning functions
│   ├── feature_engineering.py # Functions to create new features and perform encoding/scaling
│   ├── model_training.py      # Scripts for model training and hyperparameter tuning
│   ├── model_evaluation.py    # Functions to evaluate and compare model performance
│   └── utils.py               # Helper functions (e.g., logging, plotting)
├── models/
│   └── credit_risk_model.pkl  # Serialized trained model
├── api/
│   ├── app.py                 # Flask REST API to serve predictions
│   ├── requirements.txt       # API-specific dependencies
│   └── Dockerfile             # (Optional) Docker configuration for container deployment
└── requirements.txt           # Project-level Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Yitayalk/credit-risk-assessment-bati-bank.git
   cd credit-risk-assessment-bati-bank
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install API-specific dependencies:**

   ```bash
   cd api
   pip install -r requirements.txt
   cd ..
   ```

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Use the `notebooks/EDA.ipynb` notebook to:
  - Explore the dataset structure and summary statistics.
  - Visualize distributions, identify outliers, and check for missing values.
  - Analyze correlations between key numerical features.

### 2. Feature Engineering & Modeling

- Open the `notebooks/Feature_Engineering_Modeling.ipynb` notebook to:
  - Create aggregate features (e.g., total transaction amount, transaction counts).
  - Extract date-time features from transaction timestamps.
  - Encode categorical variables and scale numerical features.
  - Develop a default estimator (risk label) using RFMS formalism.
  - Train multiple ML models (Logistic Regression, Random Forest, etc.) and perform hyperparameter tuning.
  - Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

### 3. Model Deployment via REST API

- The API is built using Flask. To start the API server:
  
  ```bash
  cd api
  python app.py
  ```

- Test the API using Postman or `curl`:

  ```bash
  curl -X POST -H "Content-Type: application/json" \
       -d '{"Total_Amount": 0.5, "Avg_Amount": 0.1, "Std_Amount": 0.05, "Transaction_Count": 10, "ProviderId_encoded": 1, "ChannelId_encoded": 0}' \
       http://127.0.0.1:5000/predict
  ```

### 4. Model Management & MLOps

- Consider integrating with tools like MLflow to track experiments and manage models.
- Write unit tests and use Python logging to ensure reproducibility and easy debugging.
