
from tensorflow.keras import layers, models
from tensorflow.python.keras import Input


def DL_CNN():
    input_A = Input(shape=(28, 28, 1))
    input_B = Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(input_A)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(5, 5), activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1000, activation="relu")(x)
    x = models.Model(inputs=input_A, outputs=x)

    y = layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(input_B)
    y = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
    y = layers.Conv2D(64, kernel_size=(5, 5), activation='relu')(y)
    y = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
    y = layers.Flatten()(y)
    y = layers.Dense(1000, activation="relu")(y)
    y = models.Model(inputs=input_B, outputs=y)

    # combine the output of the two branches
    combined = layers.concatenate([x.output, y.output])

    # z = layers.Dense(1000, activation="relu")(combined)
    z = layers.Dense(10, activation="softmax")(combined)
    model = models.Model(inputs=[x.input, y.input], outputs=z)
    return model


def D_CNN(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model
