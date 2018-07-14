import pandas as pd
from keras.models import load_model

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

filepath="advanced_trained_model.best.h5"

model = load_model(filepath)
scores = model.evaluate(test_data, test_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))