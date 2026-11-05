
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Common function for both models
def show_results(model, history, X, y, name):
    loss, acc = model.evaluate(X, y, verbose=0)
    pred = model.predict(X, verbose=0).argmax(axis=1)

    print(name, "Test Loss:", round(loss, 4))
    print(name, "Test Accuracy:", round(acc, 4))
    print(classification_report(y, pred))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt='d', cmap='Blues')
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(name + " Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(name + " Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return pred

# PART 1: Feedforward Neural Network
(X, y), (Xt, yt) = keras.datasets.fashion_mnist.load_data()
print("FNN Original Shape:", X.shape, Xt.shape)

X = X.reshape(-1, 784).astype('float32') / 255
Xt = Xt.reshape(-1, 784).astype('float32') / 255
print("FNN Processed Shape:", X.shape, Xt.shape)

fnn = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

fnn.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
fnn.summary()
h1 = fnn.fit(X, y, epochs=10, validation_split=0.1)
show_results(fnn, h1, Xt, yt, "FNN")

# PART 2: Convolutional Neural Network
(X, y), (Xt, yt) = keras.datasets.mnist.load_data()
print("CNN Original Shape:", X.shape, Xt.shape)

X = X.reshape(-1, 28, 28, 1).astype('float32') / 255
Xt = Xt.reshape(-1, 28, 28, 1).astype('float32') / 255
print("CNN Processed Shape:", X.shape, Xt.shape)

cnn = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

cnn.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.summary()
h2 = cnn.fit(X, y, epochs=10, validation_split=0.1)
pred = show_results(cnn, h2, Xt, yt, "CNN")

# CNN Sample Predictions
plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(Xt[i].reshape(28, 28), cmap='gray')
    color = 'green' if yt[i] == pred[i] else 'red'
    plt.xlabel(f"True: {yt[i]}  Pred: {pred[i]}", color=color)
    plt.xticks([])
    plt.yticks([])
plt.suptitle("CNN Sample Predictions")
plt.tight_layout()
plt.show()
