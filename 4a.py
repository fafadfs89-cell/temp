from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

data = load_breast_cancer()
X1, X2, y1, y2 = train_test_split(
    data.data, data.target, test_size=0.2,
    random_state=42, stratify=data.target)

model = GridSearchCV(
    make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True)),
    {'svc__C':[0.1,1,10], 'svc__gamma':['scale',0.01,0.001]},
    cv=5, scoring='f1')

model.fit(X1, y1)
pred = model.predict(X2)

print("Best:", model.best_params_)
print(classification_report(y2, pred, digits=3))
print("Matrix:\n", confusion_matrix(y2, pred))
print("AUC:", round(roc_auc_score(y2, model.predict_proba(X2)[:,1]), 3))

RocCurveDisplay.from_estimator(model, X2, y2, name="SVM")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.grid(True)
plt.title("ROC Curve - SVM (RBF)")
plt.show()
