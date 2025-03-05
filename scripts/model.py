import pickle

# Save the best model (for example, tuned Random Forest)
with open('credit_risk_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
