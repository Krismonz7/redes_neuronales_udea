import numpy as np
from keras import layers,models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

import pandas as pd
(train_data,train_labels),(test_data,test_labels) = mnist.load_data()
train_data_flattened= train_data.reshape(train_data.shape[0],-1)

train_data_df = pd.DataFrame(train_data_flattened)

import cv2
#Apartado de cambio de tamaño de las imagenes:
# Cargar el conjunto de datos MNIST

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Tamaño al que deseas redimensionar las imágenes
new_size = (100, 100)

# Crear listas vacías para almacenar las nuevas imágenes
new_train_images = []
new_test_images = []

# Redimensionar las imágenes de entrenamiento

for image in train_images:
    resized_image = cv2.resize(image, new_size)
    new_train_images.append(resized_image)

# Redimensionar las imágenes de prueba
for image in test_images:
    resized_image = cv2.resize(image, new_size)
    new_test_images.append(resized_image)

# Convertir las listas en arreglos NumPy
new_train_images = np.array(new_train_images)
new_test_images = np.array(new_test_images)


# Comprobar las formas de los nuevos arreglos de imágenes
print("Shape of resized training images:", new_train_images.shape)
print("Shape of resized testing images:", new_test_images.shape)

model = models.Sequential()
model.add(layers.Dense(557,activation='PReLU', input_shape=(100*100,)))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='nadam',
loss='mean_squared_error',
metrics=['accuracy', 'Precision'])
x_train = new_train_images
x_train = x_train.astype("float32")/255

x_test = new_test_images.reshape((10000,100*100))

x_test = x_test.astype("float32")/255
#Aplanamos los arreglos:
x_train = new_train_images.reshape(new_train_images.shape[0], -1)
x_test = new_test_images.reshape(new_test_images.shape[0], -1)
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

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