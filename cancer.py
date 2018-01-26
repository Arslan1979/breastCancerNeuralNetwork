import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy

# seed for weight initialization
seed = 3
numpy.random.seed(seed)

df = pd.read_csv('data.csv')

# encode diagnosis malignant = 1, benign = 0
df['diagnosis'].replace({'M':1}, regex=True, inplace=True)
df['diagnosis'].replace({'B':0}, regex=True, inplace=True)
    
Y = df[df.columns[1:2]]
X = df[df.columns[2:32]]

# split dataset into train and test
x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=.66, random_state=5)

# creating model
model = Sequential()
model.add(Dense(200, input_dim = 30, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5, seed=3))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, x_test, batch_size=10, epochs=1000, validation_data=(y_train, y_test))
