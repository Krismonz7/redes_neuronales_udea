import numpy as np
from keras import layers,models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

import pandas as pd
(train_data,train_labels),(test_data,test_labels) = mnist.load_data()
train_data_flattened= train_data.reshape(train_data.shape[0],-1)

train_data_df = pd.DataFrame(train_data_flattened)

model = models.Sequential()
model.add(layers.Dense(557,activation='PReLU', input_shape=(28*28,)))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='nadam',
loss='mean_squared_error',
metrics=['accuracy', 'Precision'])
x_train = train_data_df
x_train = x_train.astype("float32")/255
print("x_train")
print(len(x_train))
x_test = test_data.reshape((10000,28*28))
x_test = x_test.astype("float32")/255

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