# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:54:09 2018

@author: zaid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from numpy import median
from sklearn.cross_validation import cross_val_score
df = pd.read_csv("rainfall in india 1901-2015.csv")
plt.style.use('ggplot')
fig = plt.figure(figsize=(18, 28))
ax = plt.subplot(2,1,1)
ax = plt.xticks(rotation=90)
ax = sns.boxplot(x='SUBDIVISION', y='ANNUAL', data=df)
ax = plt.title('Annual rainfall in all States and UT')
df[['SUBDIVISION','Jan-Feb', 'Mar-May','Jun-Sep', 'Oct-Dec']].groupby('SUBDIVISION').sum().plot.barh(stacked=True,figsize=(16,8));
regions=[]
rain={}
x={}
y={}
rgn='KERALA'
#df.set_index('SUBDIVISION',inplace=True)
df=df.dropna()
print(df['YEAR'].groupby('SUBDIVISION'))
df['JAN-FEB'] = df['JAN'] + df['FEB']
df['MAR-MAY'] = df['MAR'] + df['APR'] + df['MAY']
df['JUN-SEP'] = df['JUN'] + df['JUL'] + df['AUG'] + df['SEP']
df['OCT-DEC'] = df['OCT'] + df['NOV'] + df['DEC']
df['ANNUAL'] = df['JAN-FEB'] + df['MAR-MAY'] + df['JUN-SEP'] + df['OCT-DEC']

'''dict = {k:v for k,v in rain.groupby('SUBDIVISION')}
a1 = rain['SUBDIVISION']=='ANDAMAN & NICOBAR ISLANDS'
a2 = rain[a1]
a+string(3) = 10'''

for i in df.index:
    if i not in regions:
        regions.append(i)
        
for i in regions:
    rain[i]=df.loc[i]
    #rain[i].set_index('YEAR',inplace=True)
plt.style.use('ggplot')
fig = plt.figure(figsize=(18, 28))
ax = plt.subplot(2,1,1)
ax = plt.xticks(rotation=90)
ax = sns.boxplot(x='YEAR', y='ANNUAL', data=rain['BIHAR'])
ax = plt.title('Annual rainfall in all States and UT')     
'''for i in regions:
    rain[i].index=pd.RangeIndex(len(rain[i].index))'''
    
for i in regions:
    Q1 = rain[i]['ANNUAL'].quantile(0.25)
    Q3 = rain[i]['ANNUAL'].quantile(0.75)
    IQR = Q3-Q1
    rain[i] = rain[i][~((rain[i]['ANNUAL'] < (Q1 - 1.5 * IQR)) | (rain[i]['ANNUAL'] > (Q3 + 1.5 * IQR)))] 

 
#for i in regions:
#    rain[i].index = pd.RangeIndex(len(rain[i].index))
rain['ARUNACHAL PRADESH'][['JAN-FEB','MAR-MAY','JUN-SEP','OCT-DEC']].plot(figsize=(12,8))
plt.title("ARUNACHAL PREADESH")

