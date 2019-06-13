# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 23:57:49 2018

@author: zaid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from numpy import median
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("rainfall in india 1901-2015.csv")
regions=[]
rgn="BIHAR"
rain={}
x={}
y={}
x_train = {}
x_test = {}
y_train = {}
y_test = {}
y_pred = {}

df.set_index('SUBDIVISION',inplace=True)
df=df.interpolate()
df['JAN-MAY'] = df['JAN'] + df['FEB'] + df['MAR'] + df['APR'] + df['MAY']
df['JUN-SEP'] = df['JUN'] + df['JUL'] + df['AUG'] + df['SEP']
df['OCT-DEC'] = df['OCT'] + df['NOV'] + df['DEC']
df['ANNUAL'] = df['JAN-MAY'] + df['JUN-SEP'] + df['OCT-DEC']

for i in df.index:
    if i not in regions:
        regions.append(i)
        
for i in regions:
    rain[i]=df.loc[i]

for i in regions:
    rain[i].index = pd.RangeIndex(len(rain[i].index))
    
for i in regions:
    Q1 = rain[i]['JAN-MAY'].quantile(0.25)
    Q3 = rain[i]['JAN-MAY'].quantile(0.75)
    IQR = Q3-Q1
    rain[i] = rain[i][~((rain[i]['JAN-MAY'] < (Q1 - 1.5 * IQR)) | (rain[i]['JAN-MAY'] > (Q3 + 1.5 * IQR)))]

for i in regions:
    Q1 = rain[i]['JUN-SEP'].quantile(0.25)
    Q3 = rain[i]['JUN-SEP'].quantile(0.75)
    IQR = Q3-Q1
    rain[i] = rain[i][~((rain[i]['JUN-SEP'] < (Q1 - 1.5 * IQR)) | (rain[i]['JUN-SEP'] > (Q3 + 1.5 * IQR)))]

for i in regions:
    x[i] = np.array(rain[i].iloc[:,13].values).reshape(rain[i].iloc[:,0].values.size,1)
    y[i] = rain[i].iloc[:,14].values

for i in regions:
    for j in range(len(x[i])):
        if x[i][j] != 0:    
            x[i][j] = np.power(x[i][j],-0.5)

for i in regions:
    for j in range(len(y[i])):
        if y[i][j] != 0: 
            y[i][j] = np.power(y[i][j],-0.5)

for i in regions:
    x_train[i], x_test[i], y_train[i], y_test[i] = train_test_split(x[i], y[i], test_size = 1/3, random_state = 0)

for i in regions:
    regressor = LinearRegression()
    regressor.fit(x_train[i],y_train[i])
    y_pred[i] = regressor.predict(x_test[i])
    
for i in regions:
    for j in range(len(y_pred[i])):
        y_pred[i][j] = 1/np.power(y_pred[i][j],2)
        
for i in regions:
    for j in range(len(y_test[i])):
        y_test[i][j] = 1/np.power(y_test[i][j],2)
        

def plot_bar(i):
    # this is for plotting purpose
    index = np.arange(len(x[i]))
    plt.bar(index, y[i])
    plt.xlabel('Years', fontsize=5)
    plt.ylabel('Rainfall in mm', fontsize=5)
    plt.xticks(index, x[i], fontsize=5, rotation=30)
    plt.title("Rainfall for "+ i + " in mm from 1900-2015")
    plt.show()
    
'''for i in regions:
    plot_bar(i)'''
'''for i in regions:
    x[i] = np.array(rain[i].iloc[:,0].values).reshape(rain[i].iloc[:,0].values.size,1)
    y[i] = rain[i].iloc[:,17].values'''

#print (mean_absolute_error(y_pred[rgn],y_test[rgn]))


plt.scatter(x_train[rgn], y_train[rgn], color='red')
plt.plot(x_train[rgn], regressor.predict(x_train[rgn]),color='blue')
plt.title("Rainfall")
plt.xlabel("Rainfal in mm in JAN-MAY")
plt.ylabel("Rainfall in JUN-SEPT")
plt.show()


plt.scatter(x_train[rgn], y_train[rgn], color='red')
plt.plot(x_train[rgn], regressor.predict(x_train[rgn]),color='blue')
plt.title("Rainfall")
plt.xlabel("Rainfal in mm in JAN-MAY")
plt.ylabel("Rainfall in JUN-SEPT")
plt.show()
