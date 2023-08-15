#!/usr/bin/env python
# coding: utf-8

# # ANSWER 1(1) without centering
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\amitt\OneDrive\Desktop\Dataset.csv",header=None)
df.columns=('x','y')
x=df['x']
y=df['y']


# In[3]:


df.mean(axis=1)
array = df.to_numpy()
plt.scatter(df['x'],df['y'])
features = array.T


# In[4]:


covariance_matrix = np.cov(features)
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
m1=eig_vecs[0][1]/eig_vecs[0][0]
m2=eig_vecs[1][1]/eig_vecs[1][0]
variance2=eig_vals[0]/(eig_vals[0]+eig_vals[1])
variance1=eig_vals[1]/(eig_vals[0]+eig_vals[1])
v1="varience along PC1 is "+str(variance1*100)
v2="varience along PC2 is "+str(variance2*100)
print(v1)
print(v2)


# In[ ]:





# # Answer 1(2) with data centering

# In[5]:


var1=sum(array[:,0])/1000
var2=sum(array[:,1])/1000

array[:,0]=np.subtract(array[:,0],var1)
array[:,1]=np.subtract(array[:,1],var2)
features = array.T

covariance_matrix = np.cov(features)
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
m1=eig_vecs[0][1]/eig_vecs[0][0]
m2=eig_vecs[1][1]/eig_vecs[1][0]
variance2=eig_vals[0]/(eig_vals[0]+eig_vals[1])
variance1=eig_vals[1]/(eig_vals[0]+eig_vals[1])
v1="varience along PC1 is "+str(variance1*100)
v2="varience along PC2 is "+str(variance2*100)
print(v1)
print(v2)


# In[ ]:




