from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load train and test data
train = pd.read_csv('D:/File Pack/Courses/10Acadamey/Week 6/Technical Content/data/train.csv', low_memory=False)
test = pd.read_csv('D:/File Pack/Courses/10Acadamey/Week 6/Technical Content/data/test.csv', low_memory=False)

# Select your features and target
features = ['Total_Amount', 'Avg_Amount', 'Std_Amount', 'Transaction_Count', 'ProviderId_encoded', 'ChannelId_encoded']
X = train[features]
y = train['Risk_Label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Models: Logistic Regression:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_val)
y_prob_lr = logreg.predict_proba(X_val)[:, 1]

print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_val, y_prob_lr))

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
y_prob_rf = rf.predict_proba(X_val)[:, 1]

print("Random Forest Accuracy:", accuracy_score(y_val, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_val, y_prob_rf))

#Hyperparameter Tuning:
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_val)
y_prob_best = best_rf.predict_proba(X_val)[:, 1]
print("Tuned Random Forest ROC-AUC:", roc_auc_score(y_val, y_prob_best))
