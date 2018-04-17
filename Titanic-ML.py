
# coding: utf-8

# In[1]:


import pandas as pd
input_file1="D:/Downloads/train.csv"
input_file2="D:/Downloads/test.csv"
df1 = pd.read_csv(input_file1,header=0)
df1.head(3)
df2=pd.read_csv(input_file2,header=0)
df2.head(4)


# In[2]:


import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np
scale = StandardScaler()
y1=df1['Survived']
x1=df1[['Pclass','Sex','Age']]
x2=df2[['Pclass','Sex','Age']]
d={'male':0,'female':1}
x1['Sex']=x1['Sex'].map(d)
x2['Sex']=x2['Sex'].map(d)

x1[z]=0
z1=np.isnan(x2)
x2[z1]=0
#x1.head(100)
x1[['Pclass','Sex','Age']] = scale.fit_transform(x1[['Pclass','Sex','Age']].as_matrix())
print (x1)
x2[['Pclass','Sex','Age']] = scale.fit_transform(x2[['Pclass','Sex','Age']].as_matrix())
est1 = sm.OLS(y1, x1).fit()

est1.summary()


# In[3]:


predict=est1.predict(x2)
print(predict)
#print(y1)
for index, item in enumerate(predict):
    if (item <= 0):
        predict[index] = 0
    else :
        predict[index] = 1
        

print(predict)


# In[ ]:


predict.to_csv('D:/Downloads/output1.csv')

