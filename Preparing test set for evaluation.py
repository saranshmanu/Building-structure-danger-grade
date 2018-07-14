import numpy as np
import pandas as pd

Building_Ownership_Use = pd.read_csv('Dataset/Building_Ownership_Use.csv')
Building_Structure = pd.read_csv('Dataset/Building_Structure.csv')
Building_Structure.drop(Building_Structure.columns[0], axis=1, inplace=True)
record = pd.concat([Building_Ownership_Use, Building_Structure],axis=1)
masterReco = record

land_surface_condition = pd.get_dummies(masterReco.land_surface_condition, dummy_na=False)
foundation_type = pd.get_dummies(masterReco.foundation_type, dummy_na=False)
roof_type = pd.get_dummies(masterReco.roof_type, dummy_na=False)
ground_floor_type = pd.get_dummies(masterReco.ground_floor_type, dummy_na=False)
other_floor_type = pd.get_dummies(masterReco.other_floor_type, dummy_na=False)
position = pd.get_dummies(masterReco.position, dummy_na=False)
plan_configuration = pd.get_dummies(masterReco.plan_configuration, dummy_na=False)
legal_ownership_status = pd.get_dummies(masterReco.legal_ownership_status, dummy_na=False)
condition_post_eq = pd.get_dummies(masterReco.condition_post_eq, dummy_na=False)
col = [
    land_surface_condition,
    foundation_type,
    roof_type,
    ground_floor_type,
    other_floor_type,
    position,
    plan_configuration,
    legal_ownership_status,
    condition_post_eq
      ]
m = pd.concat(col,axis=1)
building_id = masterReco.building_id
columns_drop = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "legal_ownership_status",
    "condition_post_eq",
    "district_id",
    "vdcmun_id",
    "ward_id",
    "building_id"
      ]
masterReco.drop(columns_drop, axis=1, inplace=True)
masterReco = pd.concat([masterReco, m],axis=1)
masterReco = masterReco.astype(float)
masterReco = pd.concat([building_id, masterReco],axis=1)
masterReco.set_index("building_id", inplace=True)
masterReco.to_csv('master_reco.csv')
print(masterReco.dtypes)

df_test = pd.read_csv('Dataset/test.csv')
flag_test = masterReco.loc[[df_test.building_id[0]]]
for i in range(1, len(df_test)):
    temp_test = masterReco.loc[[df_test.building_id[i]]]
    flag_test = pd.concat([flag_test,temp_test],axis=0)
    if i%10000==0:
        print(int(i*100/len(df_test)), '% records')
print('Completed finding all the results')
flag_test = flag_test.reset_index()
saved_record_test = pd.concat([df_test, flag_test], axis=1)
saved_record_test.to_csv('advanced_test_set.csv')

df_test = pd.read_csv('advanced_test_set.csv')
columns = ['damage_grade']
district_id = df_test.district_id
district_id = pd.get_dummies(district_id, dummy_na=False)
df_test = pd.concat([df_test, district_id],axis=1)
area_assesed = df_test.area_assesed
area_assesed = df_test.get_dummies(area_assesed, dummy_na=False)
df_test = pd.concat([df_test, area_assesed],axis=1)
df_test = df_test.drop(['building_id', 'damage_grade', 'building_id.1', 'district_id', 'area_assesed', 'vdcmun_id', 'Unnamed: 0'], axis = 1)
df_test = df_test.astype(float)
df_test = df_test.fillna(0)
xnew_test = df_test

from keras.models import load_model
model = load_model('advanced_trained_model.best.h5')
ynew_test = model.predict(xnew_test)
grade = []

for i in ynew_test:
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