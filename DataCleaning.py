import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("crime.csv", encoding='latin1')
drop = df.dropna(thresh=15)
drop.to_csv("newcrime.csv")

print(drop)

cf = pd.read_csv("newcrime.csv", encoding='latin1')
indexNames = cf[(cf['Lat'] == -1) & (cf['Long'] == -1)].index
cf.drop(indexNames, inplace=True)
col = cf.dropna(thresh=16)
col.to_csv("col.csv")

print(col)

nf = pd.read_csv("col.csv", encoding='latin1')
nf.insert(6, column="crime_weight", value=1)
nf.to_csv("update_crime.csv")

