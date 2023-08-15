#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\amitt\OneDrive\Desktop\Dataset.csv",header=None)
df.columns=('x','y')
array=df.to_numpy()
plotarray=array


# In[3]:


# finding centered Kernal matrix

transposearray=np.transpose(array)
k=np.matmul(array, transposearray)
for i in range(1000):
    for j in range(1000):
        k[i][j]+=1
for i in range(1000):
    for j in range(1000):
        k[i][j]=k[i][j]*k[i][j]
        
I=np.identity(1000)

div=1/1000
divmat=np.full((1000,1000), div)
matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
#___________________________________________________________________________________________________________________________


# Finding resultant matrix

k=4
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_index=eig_vals.argsort()[-k:][::-1]
top_k_eigen=[]
for i in range(k):
    top_k_eigen.append(eig_vals[eig_index[i]])
top_k_eigen_vec=[]
for i in range(k):
    top_k_eigen_vec.append(eig_vecs[:,eig_index[i]])

top_k_eigen_vec = np.array(top_k_eigen_vec)
top_k_eigen_vec=np.transpose(top_k_eigen_vec)
array=top_k_eigen_vec
#___________________________________________________________________________________________________________________________

# initial assignment to clusters
assign=np.zeros([1000])
import random
for i in range(1000):
    assign[i]=random.randrange(k)
#____________________________________________________________________________________________________________________________   

# finding mean of clusters
mean=np.zeros([k,k])
count=np.zeros([k])
for i in range(1000):
    mean[int(assign[i])]= np.add(mean[int(assign[i])],array[i])
    count[int(assign[i])]+=1
for i in range(len(mean)):
    mean[i]=np.divide(mean[i], count[i])
#_____________________________________________________________________________________________________________________________


# In[4]:


# k-means algorithm implementation
it=0
final_error = -1
itlist=[]
errorlist=[]
while(1):
    it+=1
    itlist.append(it)
    dist=0
    error=0
    for i in range(1000):
        dist=np.linalg.norm(array[i] - mean[int(assign[i])])
        dist*=dist
        error+=dist
    errorlist.append(error)
    error_fig=plt
    if(final_error == -1):
        final_error = error
    elif(final_error > error):
        final_error = error
    else:
        break
    
    reassign=np.zeros(1000)
    dist_arr=[0]*k
    for i in range(1000):
        for j in range(k):
            dist_arr[j]=np.linalg.norm(array[i] - mean[j])
            dist_arr[j]*=dist_arr[j]
        minpos=dist_arr.index(min(dist_arr))
        if(assign[i]==minpos):
            reassign[i]=assign[i]
        else:
            reassign[i]=minpos
    #if(np.array_equal(assign, reassign)):
        #break
    #else:
    assign=reassign
    mean=np.zeros([k,k])
    count=np.zeros([k])  
    for i in range(1000):
        mean[int(assign[i])]=mean[int(assign[i])]+array[i]
        count[int(assign[i])]+=1

    for i in range(len(mean)):
        mean[i]=np.divide(mean[i], count[i])
        
#______________________________________________________________________________________________________________________________


# In[5]:



#plotting of clusters_________________________________________________________________________________________________________

colour=['red','green','blue','orange','yellow']
for i in range(1000):
    plt.scatter(plotarray[i][0], plotarray[i][1], c = colour[int(assign[i])])
#for i in range(k):
    #plt.scatter(mean[i][0], mean[i][1],marker="*",s=350,c = colour[int(assign[i])]) 
plt.show()

#______________________________________________________________________________________________________________________________


# In[ ]:




