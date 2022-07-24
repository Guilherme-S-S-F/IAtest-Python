import keras
import numpy as np
from keras import layers
from keras import losses

def GiveData(data_size):
    array = []
    for n in range(0,data_size):
        x = np.random.randint(0, 64)
        y = np.random.randint(0,64)
        array.append([[x,y]])
    
    return array

def GetAnswer(data):
    array = []

    for n in data:
        if n[0][0] > 30:
            array.append([[1]])
        else:
            array.append([[0]])

    return array

def StartTrain():
    network = keras.Sequential()

    network.add(layers.Input(shape=(None,2)))
    network.add(layers.Dense(12, activation='relu'))
    network.add(layers.Dense(24, activation='relu'))
    network.add(layers.Dense(2, activation='softmax'))

    network.compile(optimizer='adam',
    loss= losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

    x_array = GiveData(300)
    y_array = GetAnswer(x_array)

    x_train = np.array(x_array)

    y_train = np.array(y_array)

    network.fit(x=x_train, y=y_train, batch_size=32, epochs=400)
    network.save(save_format="tf", filepath="IAdata/")

