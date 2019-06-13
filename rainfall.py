import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from numpy import median
from sklearn.cross_validation import cross_val_score
df = pd.read_csv("rainfall in india 1901-2015.csv")         #raw dataset
regions=[]
rain={}
x={}
y={}
rgn='COASTAL KARNATAKA' #Select a state/region
df.set_index('SUBDIVISION',inplace=True)
df=df.dropna()     #for removing tuples with na values

#Different seasons 
df['JAN-FEB'] = df['JAN'] + df['FEB']
df['MAR-MAY'] = df['MAR'] + df['APR'] + df['MAY']                
df['JUN-SEP'] = df['JUN'] + df['JUL'] + df['AUG'] + df['SEP']
df['OCT-DEC'] = df['OCT'] + df['NOV'] + df['DEC']
df['ANNUAL'] = df['JAN-FEB'] + df['MAR-MAY'] + df['JUN-SEP'] + df['OCT-DEC']

#extracting all the name of the regions present in the dataset
for i in df.index:
    if i not in regions:
        regions.append(i)
        
for i in regions:
    rain[i]=df.loc[i]
    
plt.scatter(rain[rgn]['YEAR'].values, np.power(rain[rgn]['ANNUAL'].values,-0.5), color='red')
plt.title("Rainfall in Coastal Karnataka")
plt.xlabel("Year")
plt.ylabel("Rainfall in mm")
plt.show() 
for i in regions:
    rain[i].index = pd.RangeIndex(len(rain[i].index))
    
   

#Outlier analysis using IQR  
for i in regions:
    Q1 = rain[i]['ANNUAL'].quantile(0.25)
    Q3 = rain[i]['ANNUAL'].quantile(0.75)
    IQR = Q3-Q1
    rain[i] = rain[i][~((rain[i]['ANNUAL'] < (Q1 - 1.5 * IQR)) | (rain[i]['ANNUAL'] > (Q3 + 1.5 * IQR)))]

for i in regions:
    x[i] = np.array(rain[i]['YEAR'].values).reshape(rain[i]['YEAR'].size,1)
    y[i] = np.power(rain[i]['ANNUAL'].values,-0.5)               #Scaling down using inverse square root

#splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
x_train={}
y_train={}
x_test={}
y_test={}
for i in regions:
   x_train[i],x_test[i],y_train[i],y_test[i] = train_test_split(x[i],y[i],test_size=0.25,random_state=0)   #25% data is used for testing  


y_pred={}
y_pred1={}

#applying linear regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
for i in regions:
    regressor=LinearRegression()
    sv=SVR(kernel='rbf')
    regressor.fit(x_train[i],y_train[i])
    sv.fit(x_train[i],y_train[i])
    y_pred[i]= regressor.predict(x_test[i])
    y_pred1[i]=sv.predict(x_test[i])

for i in regions:
    lreg = LinearRegression()
    lin_score= cross_val_score(lreg,np.array(x[i]),np.array(y[i]),cv=10)
    print(i,-lin_score.mean())

'''acc=np.zeros(len(regions))
for i in regions:
    acc=0
    for j in range(len(y_test[i])):
        acc+=(y_test[i][j]-np.abs(y_test[i][j]-y_pred[i][j]))/y_test[i][j]
    print(i,acc/len(y_test[i]))'''
      
'''regressor=LinearRegression()
regressor.fit(x_train[rgn],y_train[rgn])    
plt.scatter(x[rgn], y[rgn], color='red')
plt.title("Rainfall")
plt.xlabel("Year")
plt.ylabel("Rainfall in mm")
plt.show()

plt.scatter(x_test[rgn], y_test[rgn], color='red')
plt.plot(x_test[rgn], regressor.predict(x_test[rgn]),color='blue')
plt.title("Rainfall")
plt.xlabel("Year")
plt.ylabel("Rainfall in mm")
'''
'''plt.scatter(x_train[rgn], y_train[rgn], color='red')
plt.plot(x_train[rgn], sv.predict(x_train[rgn]),color='blue')
plt.title("Rainfall")
plt.xlabel("Year")
plt.ylabel("Rainfall in mm")
plt.show()

plt.scatter(x_test[rgn], y_test[rgn], color='red')
plt.plot(x_test[rgn], sv.predict(x_test[rgn]),color='blue')
plt.title("Rainfall")
plt.xlabel("Year")
plt.ylabel("Rainfall in mm")
plt.show()'''
