from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

import numpy as np
X = np.random.rand(1000,10)
y = np.random.choice([1,0], size=1000)

model=models.Sequential()
layer1=layers.Dense(10,activation="relu")

model.add(layer1)

layer2=layers.Dense(50,activation="relu")
#

model.add(layer2)
model.add(layers.Reshape((1,50)))

layer5=layers.LSTM(50,activation="relu")
model.add(layer5)
layer3=layers.Dense(20,activation="relu")
#
model.add(layer3)
layer4=layers.Dense(1,activation="softmax")
model.build(input_shape=X.shape)
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
print(model.summary())
print("Layer 1 input shape:",layer1.input_shape, " Output shape:",layer1.output_shape)
print("Layer 2 input shape:",layer2.input_shape, " Output shape:",layer2.output_shape)
print("Layer 3 input shape:",layer3.input_shape, " Output shape:",layer3.output_shape)
print("Layer 5 input shape:",layer5.input_shape, " Output shape:",layer5.output_shape)
hist=model.fit(X,y,batch_size=10,epochs=10)
