from keras import layers, models
model = models.Sequential()
# Capa convolucional
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))


# Capa de maxpooling
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# Capa de aplanamiento
model.add(layers.Flatten())


# Capas densas
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 clases en MNIST

# Compilación del modelo
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
# Resumen del modelo
model.summary()
