import numpy as np
from keras import layers,models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

import pandas as pd
#Descomponemos la data que entrega por default mnist en tran y test
#lo separamos en data y labels
(train_data,train_labels),(test_data,test_labels) = mnist.load_data()
#Reshape es el proceso para aplanar las imagenes y convertirlas en una data 2d

train_data_flattened= train_data.reshape(train_data.shape[0],-1)
#Lo volvemos una dataframe con la libreria pandas
train_data_df = pd.DataFrame(train_data_flattened)


#Mostramos las primeras diez registros o rows del dataframe con un .head de pandas
#print(train_data_df.head(7))
#Mostramos la data de la dataframe que acabamos de crear
#print(train_data_df.shape)

#Usamos matplotlib paragraficar el dataframe y ver que imagen forma el dataframe de entrenamiento
# para las redes neuronales :
plt.imshow(train_data[797])

#plt.show()
#El numero que corresponde a la imagen mostrada arriba
#print(train_labels[797])

#Creamos un nuevo modelo vacio en Keras
model = models.Sequential()

#Comando para agregar una capa de 512 nuronas al modelo
#con funcion de activacion relu

model.add(layers.Dense(512,activation='relu', input_shape=(28*28,)))

#Comando para agregar una capa de 10 neuronas
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy', 'Precision'])


#Imprimimos el estado del modelo:
#print(model.summary())

#Ajustando los parametros de la x
#Ahora asignamos los valores a unavariable:
x_train = train_data_df
#Convierte los valores de la data a tipo flotante(float32) 
x_train = x_train.astype("float32")/255
x_test = test_data.reshape((10000,28*28))

x_test = x_test.astype("float32")/255

#print(x_train[0])

#Ahora realuzamos el proceso con la y:


y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

print(y_train[1])
#to_categorical: esta funcion keras toma un conjunto de etiquetas y los convierte a 
# un formato numerico , en un hot-one, por ejemplo: ["perro","gato","ave"]:
# perro = [1,0,0]
# gato = [0,1,0]
# ave = [0,0,1]
#print(y_train[0])
#print(y_test[0])

from keras.callbacks import TensorBoard

#seusa para almacenar lainformaion del modelo,curvas de aprendizaje etc.. 
#las almacena en la direccion especificada logs/denso
tensorboardDenso = TensorBoard(log_dir="logs/denso")

#Historial

history = model.fit(x_train,y_train,epochs=20,callbacks=[tensorboardDenso],batch_size=128,validation_data=(x_test,y_test))

#x_train: son los datos que entran a la red
#y_train: son los valores asociados a los datos de entrada (x_train),
#son los valores que el modelo intenta predecir
#epochs: es lacantidad de veces que el modelo recorre el conjunto de datos
#durante el entrenamiento, cada epoch es una pasada hacia adelante y hacia atras

#batch_size: el entrenamiento separa los datos que se entregan al modelo 
# en lotes, por lo que estos tienen un tamaño, para eso es batch_size, indica el peso 
# de cada lote de datos para ahorrar memoria
#validation_data: Son los resultados que esperan ser logrados 

#Callbacks: son funciones que se ejecutan en varios puntos del codigo et permite varias 
# opciones, como registrar metricas, almacenar checkpoints, etc...

pd.DataFrame({"loss":history.history["loss"] , 
              "val_loss":history.history["val_loss"]}).plot(figsize=(7,7))


#Esta funcion le agrega una cuadricula al plot
plt.grid(True)

plt.axis
plt.xlabel("Epoch")
plt.ylabel("Y - label")
plt.show()
print("\n\n Evaluacion comparada con los test: \n\n")
model.evaluate(x_test,y_test)
#Loss: Representa la perdida en el uso de datos de !entrenamiento
#val_loss: Representa la perdida en el conjunto de datos de !validacion
#Figsize: Se usa como en css se usa el width y height, son para indicar el 
# espacio qure va a ocupar la grafica