import tensorflow as tf

print("TF Version:", tf.__version__)
print("Eager Execution:", tf.executing_eagerly())

# Tensor creation
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.ones((2, 2))
c = tf.random.uniform((2, 2), 0, 10)

# Basic operations
print("a + b:\n", a + b)
print("a * c:\n", a * c)
print("a @ b:\n", a @ b)

# Manipulation
print("Reshape a:\n", tf.reshape(a, (4, 1)))
print("Slice a[0]:", a[0])

# Computation graph with tf.function
@tf.function
def compute(x, y):
    return tf.sqrt(x**2 + y**2)

print("Graph result:\n", compute(a, b))

# Eager execution example
for i in range(2):
    x = tf.random.uniform((2, 2))
    print(f"Run {i+1}, mean:", tf.reduce_mean(x).numpy())
