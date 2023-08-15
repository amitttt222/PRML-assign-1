#!/usr/bin/env python
# coding: utf-8

# # importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Finding Kernal matrix

# In[2]:


def find_kernal_matrix(array):
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
    return kernalcent


# # Finding top k eigen vectors

# In[3]:



def find_H_matrix(kernalcent,k):
    eig_vals, eig_vecs = np.linalg.eig(kernalcent)
    #eig_vecs = eig_vecs.transpose()
    #eig_vecs.shape
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
    return array


# # Assignment of data to clusters

# In[4]:


def assignment_to_cluster(array,k):
    assign=np.zeros([1000])
    for i in range(1000):
        index=0
        maximum=array[i][0]
        for j in range(1,k):
            if(array[i][j]>maximum):
                index=j
                maximum=array[i][j]
        assign[i]=index
    return assign
  


# # Finding mean of Clusters

# In[5]:


def find_mean_of_cluster(arrayplot,assign):
    mean=np.zeros([k,2])
    count=np.zeros([k])
    for i in range(1000):
        mean[int(assign[i])]= np.add(mean[int(assign[i])],arrayplot[i])
        count[int(assign[i])]+=1
    for i in range(len(mean)):
        mean[i]=np.divide(mean[i], count[i])
    return mean


# # Plotting the clusters

# In[6]:



def plot(arrayplot,assign,mean):
    colour=['red','green','blue','orange','yellow']
    for i in range(1000):
        plt.scatter(arrayplot[i][0], arrayplot[i][1], c = colour[int(assign[i])])
    for i in range(k):
        plt.scatter(mean[i][0], mean[i][1],c = colour[i],marker='*',s=350)  
    plt.show()


# # Reading the CSV file

# In[7]:


df=pd.read_csv(r"C:\Users\amitt\OneDrive\Desktop\Dataset.csv",header=None)
df.columns=('x','y')


# # Converting dataframe to numby array

# In[8]:


array=df.to_numpy()
arrayplot=array
plotarray=array

number_of_clusters=4
k=number_of_clusters


# # Plotting of clusters

# In[9]:


kernalcent=find_kernal_matrix(array)
array=find_H_matrix(kernalcent,k)
assign=assignment_to_cluster(array,k)
mean=find_mean_of_cluster(arrayplot,assign)
plot(arrayplot,assign,mean)


# In[ ]:





# In[ ]:




