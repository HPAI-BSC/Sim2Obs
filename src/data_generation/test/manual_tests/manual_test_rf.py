import joblib
from sklearn.tree import export_text

model_path = '../../model/Best_Model-Random-3par.pkl'
model = joblib.load(model_path)
print(model.get_params())
importances = model.feature_importances_
print(importances)

# Check if the model has OOB score enabled
if hasattr(model, 'oob_score_'):
    oob_error = 1 - model.oob_score_
    print(f"OOB Error: {oob_error * 100:.2f}%")

tree_0 = model.estimators_[0]
tree_rules = export_text(tree_0)
print(tree_rules)
