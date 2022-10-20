import numpy as np
import tensorflow as tf

# linear binary output with MSE
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(units=10, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# a circle with r=1 and center=(0,0)
x = np.array([
    (1.1, 0.99), (1.03, 0.78), (2, 2), (1.23, 1.11),
    (0.63, 0.7), (0.4, 0.9), (0.8, 0.9), (0.3, 0.2), (0.8, 0.8), (0.11, 0.93), (0.5, 0.5), (0.9, 0.11),
    (0.89, 1.3), (0.44, 1.2), (0.34, 1.11), (0.88, 1.2), (1.3, 1.3), (2.0, 1.0)])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=500)

preds = model.predict([
    (1.09, 0.95),
    (0.4, 0.98),
    (0.1, 0.1),
    (1.2, 1.4)
]);

correct = [1, 0, 0, 1]
result = [[pred[0], int(pred[0] > 0.5)] for pred in preds]
total = 0
for i, c in enumerate(correct):
    if c == result[i][1]:
        total += 1
print(result)
print("Correct:", total, "/", len(correct))