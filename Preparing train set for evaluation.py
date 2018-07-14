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

df = pd.read_csv('Dataset/train.csv')
flag = masterReco.loc[[df.building_id[0]]]
for i in range(1, len(df)):
    temp = masterReco.loc[[df.building_id[i]]]
    flag = pd.concat([flag,temp],axis=0)
    if i%10000==0:
        print(int(i*100/len(df)), '% records')
print('Completed finding all the results')
flag = flag.reset_index()
saved_record = pd.concat([df, flag], axis=1)
saved_record.to_csv('advanced_train_set.csv')