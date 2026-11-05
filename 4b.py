from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import RocCurveDisplay
import pandas as pd
import matplotlib.pyplot as plt

data = load_breast_cancer()
X1, X2, y1, y2 = train_test_split(
    data.data, data.target, test_size=0.2,
    random_state=42, stratify=data.target)

model = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {'n_estimators':[50,100], 'max_depth':[None,10]},
    cv=3, scoring='f1')

model.fit(X1, y1)
pred = model.predict(X2)

print("Best:", model.best_params_)
print(classification_report(y2, pred, digits=3))
print("Matrix:\n", confusion_matrix(y2, pred))
print("AUC:", round(roc_auc_score(y2, model.predict_proba(X2)[:,1]), 3))

pd.Series(model.best_estimator_.feature_importances_,
          index=data.feature_names).nlargest(5).plot.barh()
plt.title("Top 5 Features")
plt.show()


RocCurveDisplay.from_estimator(model, X2, y2, name="Random Forest")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.grid(True)
plt.title("ROC Curve - Random Forest")
plt.show()
