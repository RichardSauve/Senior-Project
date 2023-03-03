import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

data = pd.read_excel('games_data.xlsx')
data = data.dropna()
y = data['Win']
X = data[['PTS%', 'SRS', 'SOS', 'GF/G', 'GA/G', 'PP%', 'PK%', 'SV%', 'S%']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Dense(128, activation='relu', input_dim=9))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

print(model.predict(X_test))

score = model.evaluate(X_test, y_test)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

