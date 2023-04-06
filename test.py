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

data = pd.read_excel('TOR.xlsx', header=0, index_col=0)

X = data.drop(['Date', 'AT', 'Opponent', 'GF', 'GA', 'OT', 'Result'], axis = 1).to_numpy()
y = data['Result'].to_numpy()
win = y == 'W'
y = np.where(win, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=88, train_size=0.2)

models = {}


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

models['Logistic Regression'] = LogisticRegression()
models['Support Vector Machines'] = LinearSVC()
models['Decision Trees'] = DecisionTreeClassifier()
models['Random Forest'] = RandomForestClassifier()
models['Naive Bayes'] = GaussianNB()
models['K-Nearest Neighbor'] = KNeighborsClassifier()

accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    # Fit the classifier model
    models[key].fit(X_train, y_train)

    # Prediction
    predictions = models[key].predict(X_test)

    # Calculate Accuracy, Precision and Recall Metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)

for mod, score in accuracy.items():
    print(mod, score)

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


