import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Common function
def run(model, X, y, Xt, yt, name):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    h = model.fit(X, y, epochs=10, validation_split=0.1)
    loss, acc = model.evaluate(Xt, yt, verbose=0)
    pred = model.predict(Xt, verbose=0).argmax(axis=1)

    print(name, "Accuracy:", acc)
    print(classification_report(yt, pred, zero_division=0))

    sns.heatmap(confusion_matrix(yt, pred), annot=True, fmt='d', cmap='Blues')
    plt.title(name + " Confusion Matrix")
    plt.show()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(h.history['accuracy'], label='Train')
    plt.plot(h.history['val_accuracy'], label='Validation')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(h.history['loss'], label='Train')
    plt.plot(h.history['val_loss'], label='Validation')
    plt.title("Loss")
    plt.legend()
    plt.show()

    return pred

# FNN
(X,y),(Xt,yt) = keras.datasets.fashion_mnist.load_data()
X = X.reshape(-1,784).astype('float32')/255
Xt = Xt.reshape(-1,784).astype('float32')/255

fnn = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

fnn.summary()
run(fnn, X, y, Xt, yt, "FNN")

# CNN
(X,y),(Xt,yt) = keras.datasets.mnist.load_data()
X = X.reshape(-1,28,28,1).astype('float32')/255
Xt = Xt.reshape(-1,28,28,1).astype('float32')/255

cnn = keras.Sequential([
    keras.Input(shape=(28,28,1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

cnn.summary()
pred = run(cnn, X, y, Xt, yt, "CNN")

# Sample predictions
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(Xt[i].reshape(28,28), cmap='gray')
    plt.xlabel(f"True:{yt[i]} Pred:{pred[i]}",
               color='green' if yt[i]==pred[i] else 'red')
    plt.xticks([])
    plt.yticks([])
plt.show()
