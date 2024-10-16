# install pandas, 

import pandas as pd
import numpy as np
import os
import math
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
import matplotlib.pyplot as plt
import geopy.distance as dis
from patsy import dmatrix, dmatrices

cwd = os.getcwd()
print(cwd)
data =  pd.read_csv("./data/tt_data_for_ml.csv")

# data.columns
# data.isna().any()
# data.head()
# data.describe()
# data.dtypes

data = data.dropna()
data = data[["t_travtime", "lat_d", "lon_d", "lat_o", "lon_o", "od_dist", "year"]]
data.head()

def calcdis(row):
    point_o = (row['lat_o'], row['lon_o'])
    point_d = (row['lat_d'], row['lon_d'])
    distpp = dis.distance(point_o, point_d).km
    return distpp

def centerdiso(row, center):
    point_o = (row['lat_o'], row['lon_o'])
    distpp = dis.distance(point_o, center).km
    return distpp

def centerdisd(row, center):
    point_d = (row['lat_d'], row['lon_d'])
    distpp = dis.distance(center, point_d).km
    return distpp

def mean_error(y, y_pred):
    return np.mean(y_pred-y)

#aa = dis2(data.loc[[2]])
zocalo = [19.432777,-99.133217]
data['od_dist2'] = data.apply(calcdis, axis=1)
data['resdis'] = data.apply(centerdiso, center=zocalo, axis=1)
data['desdis'] = data.apply(centerdisd, center=zocalo, axis=1)

# testing dist measures
data_ols1 = data[["t_travtime","od_dist", "resdis", "desdis"]]
data_ols2 = data[["t_travtime","od_dist2", "resdis", "desdis"]]

y1 = data_ols1["t_travtime"]
X1 = data_ols1.drop(["t_travtime"], axis = 1)

y2 = data_ols2["t_travtime"]
X2 = data_ols2.drop(["t_travtime"], axis = 1)

# original distance
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
model1 = LinearRegression()
model1.fit(X_train1, y_train1)
y_pred1 = model1.predict(X_test1)
rmse1 = root_mean_squared_error(y_test1, y_pred1)
r21 = r2_score(y_test1, y_pred1)
mae1 = mean_absolute_error(y_test1, y_pred1)
mape1 = mean_absolute_percentage_error(y_test1, y_pred1)
mde1 = median_absolute_error(y_test1, y_pred1)
me1 = mean_error(y_test1, y_pred1)
print(f'RMSE: {rmse1}') 
print(f'R² Score: {r21}')
print(f'Mean Abs Err: {mae1}')
print(f'Mean Abs % Err: {mape1}')
print(f'Med Abs Err: {mde1}')
print(f'Mean Err: {me1}')

# updated distance
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = LinearRegression()
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)
rmse2 = root_mean_squared_error(y_test2, y_pred2)
r22 = r2_score(y_test2, y_pred2)
mae2 = mean_absolute_error(y_test2, y_pred2)
mape2 = mean_absolute_percentage_error(y_test2, y_pred2)
mde2 = median_absolute_error(y_test2, y_pred2)
me2 = mean_error(y_test2, y_pred2)
print(f'RMSE: {rmse2}') 
print(f'R² Score: {r22}')
print(f'Mean Abs Err: {mae2}')
print(f'Mean Abs % Err: {mape2}')
print(f'Med Abs Err: {mde2}')
print(f'Mean Err: {me2}')

# main OLS specification

y_ols, X_ols = dmatrices('t_travtime ~ od_dist2:C(year) + resdis:C(year) + desdis:C(year) + C(year)', data)
X_train, X_test, y_train, y_test = train_test_split(X_ols, y_ols, test_size=0.2, random_state=42)
olsmod = LinearRegression()
olsmod.fit(X_train, y_train)
y_pred = olsmod.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mde = median_absolute_error(y_test, y_pred)
me = mean_error(y_test, y_pred)
print(f'RMSE: {rmse}') 
print(f'R² Score: {r2}')
print(f'Mean Abs Err: {mae}')
print(f'Mean Abs % Err: {mape}')
print(f'Med Abs Err: {mde}')
print(f'Mean Err: {me}')