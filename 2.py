
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

X1, X2, y1, y2 = train_test_split(X, y, test_size=0.3, random_state=42)

# SVM
svm = SVC(kernel='linear')
svm.fit(X1, y1)
p1 = svm.predict(X2)
print("SVM Accuracy:", accuracy_score(y2, p1))
print("SVM Confusion Matrix:\n", confusion_matrix(y2, p1))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X1, y1)
p2 = rf.predict(X2)
print("Random Forest Accuracy:", accuracy_score(y2, p2))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y2, p2))

# Heatmaps
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y2, p1), annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y2, p2), annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()
