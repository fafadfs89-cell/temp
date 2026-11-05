
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X1, X2, y1, y2 = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion='gini', random_state=0)
model.fit(X1, y1)
pred = model.predict(X2)

cm = confusion_matrix(y2, pred)
print("Confusion Matrix:\n", cm)
print("Accuracy Score:", accuracy_score(y2, pred))

# Confusion Matrix Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Clear Decision Tree
plt.figure(figsize=(18, 15))
plot_tree(model, filled=True, feature_names=iris.feature_names,
          class_names=iris.target_names, fontsize=9)
plt.title("Decision Tree Visualization")
plt.show()

plt.figure(figsize=(18,15))
plot_tree(model, filled=True)
plt.show()
