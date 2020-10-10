import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras import layers

import data

# Generating the data
train_validation = 1/4

ti_train = data.e2_ti[0:int(np.round(train_validation*len(data.e2_ti)))]
te_train = data.e2_te[0:int(np.round(train_validation*len(data.e2_ti)))]
q_train = data.e2_q[0:int(np.round(train_validation*len(data.e2_ti)))]


# Making the model
tf.keras.backend.set_floatx('float64')
merged_array = np.stack([ti_train, te_train], axis=1)

input_shape = merged_array.shape
target_shape = q_train.shape

model = Sequential()

model.add(layers.Dense(3, input_dim=2, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1))

model.compile(loss='mse', optimizer='nadam')

model.fit(merged_array, q_train, epochs=15000)

model.summary()
model.get_config()

filepath = './saved_model_e2_1_4'
save_model(model, filepath)
