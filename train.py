import tensorflow as tf
from tensorflow.keras import layers
import json
import os
import time

MODEL_NAME = 'model'

start_time = time.perf_counter()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = tf.keras.models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

checkpoint_path = os.path.join('ckpt', 'cp-{epoch:02d}.keras')
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=False, 
    save_freq='epoch'       
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
        x_train,
        y_train,
        epochs=20, 
        callbacks=[cp_callback],
        validation_data=(x_test, y_test)
    )

model.save(MODEL_NAME + '.keras')
with open(MODEL_NAME + '.json', 'w') as file:
    json.dump(history.history, file, indent=4)


end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

