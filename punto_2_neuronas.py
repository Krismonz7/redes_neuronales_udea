import numpy as np
from keras import layers, models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

model = models.Sequential()
# Capa convolucional
model.add(layers.Conv2D(10, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())

# Capa de maxpooling
model.add(layers.MaxPool2D(pool_size=(2, 2)))
# Capa de aplanamiento
model.add(layers.Flatten())
# Capas densas
model.add(layers.Dense(557, activation='PReLU'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='nadam',
              loss='mean_squared_error',
              metrics=['accuracy', 'Precision'])

x_train = train_data.reshape(train_data.shape[0], 28, 28, 1)
x_train = x_train.astype("float32") / 255

x_test = test_data.reshape(test_data.shape[0], 28, 28, 1)
x_test = x_test.astype("float32") / 255

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# Resto del código...


from keras.callbacks import TensorBoard
tensorboardDenso = TensorBoard(log_dir="logs/denso")

history = model.fit(x_train,y_train,epochs=20,callbacks=[tensorboardDenso],
                    
                    batch_size=250,validation_data=(x_test,y_test))
pd.DataFrame({"loss":history.history["loss"] ,
              "val_loss":history.history["val_loss"]}).plot(figsize=(10,7))

plt.grid(True)
plt.xlabel("Epoch")
plt.ylabel("Y - results")
plt.show()
print("\n\n Evaluacion comparada con los test: \n\n")
model.evaluate(x_test,y_test)
print(model.summary())