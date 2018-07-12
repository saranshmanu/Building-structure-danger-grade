import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
model.add(Dense(120, input_dim=120, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dropout(0.6))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

df = pd.read_csv('advanced_train_set.csv')
columns = ['damage_grade']
y = df.damage_grade
y = pd.get_dummies(y, dummy_na=False)

district_id = df.district_id
district_id = pd.get_dummies(district_id, dummy_na=False)
df = pd.concat([df, district_id],axis=1)
area_assesed = df.area_assesed
area_assesed = pd.get_dummies(area_assesed, dummy_na=False)
df = pd.concat([df, area_assesed],axis=1)
df = df.drop(['building_id', 'damage_grade', 'building_id.1', 'district_id', 'area_assesed', 'vdcmun_id', 'Unnamed: 0'], axis = 1)
df = df.astype(float)
df = df.fillna(0)
x = df

train_data = x[:500000]
train_labels = y[:500000]
test_data = x[500000:]
test_labels = y[500000:]
print(train_data[:5])

model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=1000, batch_size=100)
scores = model.evaluate(test_data, test_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
model.save('advanced_trained_model.h5')