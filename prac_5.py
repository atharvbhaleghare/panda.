import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess MNIST
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_train = np.expand_dims(x_train, -1).astype("float32")

# Create a simple random generator (for demo only)
noise = tf.random.normal((16, 28, 28, 1))
generated = noise  # since we don’t have pretrained model

# Display random "generated" images
plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(generated[i,:,:,0], cmap='gray')
    plt.axis('off')
plt.show()