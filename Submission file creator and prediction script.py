import pandas as pd
import numpy as np

df = pd.read_csv('advanced_test_set.csv')
district_id = df.district_id
district_id = pd.get_dummies(district_id, dummy_na=False)
df = pd.concat([df, district_id],axis=1)
area_assesed = df.area_assesed
area_assesed = pd.get_dummies(area_assesed, dummy_na=False)
df = pd.concat([df, area_assesed],axis=1)
df = df.drop(['building_id', 'building_id.1', 'district_id', 'area_assesed', 'vdcmun_id', 'Unnamed: 0'], axis = 1)
df = df.astype(float)
df = df.fillna(0)
x = df

from keras.models import load_model
model = load_model('advanced_trained_model.best.h5')
y = model.predict(x)
grade = []

for i in y:
    ind = np.unravel_index(np.argmax(i, axis=None), i.shape)
    if ind == (0, ):
        grade.append('Grade 1')
    elif ind == (1, ):
        grade.append('Grade 2')
    elif ind == (2, ):
        grade.append('Grade 3')
    elif ind == (3, ):
        grade.append('Grade 4')
    elif ind == (4, ):
        grade.append('Grade 5')
print(len(grade))

df = pd.read_csv('Dataset/test.csv')
building_id = df.building_id
grade_id = pd.DataFrame(grade)
result = pd.concat([building_id, grade_id],axis=1)
result = result.rename(columns={0: 'damage_grade'})
print(result[:10])
result.to_csv('submission_advanced.csv',mode = 'w', index=False)