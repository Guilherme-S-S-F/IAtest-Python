from time import sleep
import keras
import os
import train
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import losses

if not os. path. isdir(s= "IAdata/"):
    print("No data founded...")
    sleep(3)
    print("creating new data...")
    sleep(3)
    train.StartTrain()

print("Loading Example...")
sleep(3)

network= keras.models.load_model(filepath="IAdata/")


def GiveTrainData(data_size):
    array = []
    for n in range(0,data_size):
        x = np.random.randint(0, 64)
        y = np.random.randint(0,64)
        array.append([[x,y]])
    
    return array


data1 = GiveTrainData(100)

graph = plt
x1 = []
y1 = []

x2 = []
y2 = []

for n in data1:
    result = network.predict(x=n)
    
    if result[0][0] > 0.5:
        x1.append(n[0][0])
        y1.append(n[0][1])
    else:
        x2.append(n[0][0])
        y2.append(n[0][1])

graph.plot(x1, y1, color='none', marker='o', markersize=6, markerfacecolor='blue', markeredgecolor='blue')
graph.plot(x2, y2, color='none', marker='o', markersize=6, markerfacecolor='red', markeredgecolor='red')
graph.title("Red = numbers > 30")
graph.grid()
graph.show()
