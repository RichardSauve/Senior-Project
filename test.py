import datetime

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
import matplotlib.pyplot as plt
from itertools import compress
from sklearn.metrics import roc_curve, auc

# data = pd.read_excel('games_data.xlsx')
# data = data.dropna()
# y = data['Win']
# X = data[['PTS%', 'SRS', 'SOS', 'GF/G', 'GA/G', 'PP%', 'PK%', 'SV%', 'S%']]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=91)
#
# model = Sequential()
# model.add(Dense(50, activation='relu', input_dim=9))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_data=(X_test, y_test))
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
# score = model.evaluate(X_test, y_test)
# print(f"Test loss: {score[0]}")
# print(f"Test accuracy: {score[1]}")

timenow = pd.to_datetime(datetime.datetime.now(), format='%Y-%m-%d') - pd.to_datetime('2022-10-06', format='%Y-%m-%d')


print(timenow)

