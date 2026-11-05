import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = x_train.reshape(-1, 28, 28, 1)

batch_size = 256
latent_dim = 100

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(60000).batch(batch_size)

# Generator
generator = tf.keras.Sequential([
    layers.Dense(7*7*128, input_shape=(100,)),
    layers.Reshape((7, 7, 128)),
    layers.Conv2DTranspose(64, 4, strides=2, padding='same'),
    layers.LeakyReLU(),
    layers.Conv2DTranspose(1, 4, strides=2,
                           padding='same', activation='tanh')
])

# Discriminator
discriminator = tf.keras.Sequential([
    layers.Conv2D(64, 4, strides=2, padding='same',
                  input_shape=(28, 28, 1)),
    layers.LeakyReLU(),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

# Loss and optimizers
loss_fn = tf.keras.losses.BinaryCrossentropy()

g_opt = tf.keras.optimizers.Adam(0.0001)
d_opt = tf.keras.optimizers.Adam(0.0001)

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([tf.shape(images)[0], latent_dim])

    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake = generator(noise, training=True)

        real_out = discriminator(images, training=True)
        fake_out = discriminator(fake, training=True)

        g_loss = loss_fn(tf.ones_like(fake_out), fake_out)
        d_loss = (loss_fn(tf.ones_like(real_out), real_out) +
                  loss_fn(tf.zeros_like(fake_out), fake_out))

    g_grad = gt.gradient(g_loss, generator.trainable_variables)
    d_grad = dt.gradient(d_loss, discriminator.trainable_variables)

    g_opt.apply_gradients(zip(g_grad, generator.trainable_variables))
    d_opt.apply_gradients(zip(d_grad, discriminator.trainable_variables))

    return g_loss, d_loss

# Train GAN
for epoch in range(100):
    for images in dataset:
        g_loss, d_loss = train_step(images)

    print("Epoch:", epoch+1,
          "Generator Loss:", float(g_loss),
          "Discriminator Loss:", float(d_loss))

# Generate images
noise = tf.random.normal([16, latent_dim])
images = generator(noise, training=False)

plt.figure(figsize=(5, 5))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow((images[i, :, :, 0] + 1) / 2, cmap='gray')
    plt.axis('off')

plt.show()

print("GAN Training Completed!")
