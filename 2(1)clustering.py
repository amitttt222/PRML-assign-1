#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


# # Reading CSV File

# In[2]:


df=pd.read_csv(r"C:\Users\amitt\OneDrive\Desktop\Dataset.csv",header=None)
df.columns=('x','y')
array=df.to_numpy()


# # Initial assignment of datapoint to cluster 

# In[3]:


def initial_assignment(k):
    assign=np.zeros([1000])
    import random
    for i in range(1000):
        assign[i]=random.randrange(k)
    seta=set(assign)
    while(len(seta)!=k):
        for i in range(1000):
            assign[i]=random.randrange(k)
    return assign


# # Finding Initial mean

# In[4]:



def find_initial_mean(k,assign):
    mean=np.zeros([k,2])
    count=np.zeros([k])
    for i in range(1000):
        mean[int(assign[i])]= np.add(mean[int(assign[i])],array[i])
        count[int(assign[i])]+=1
    for i in range(len(mean)):
        mean[i]=np.divide(mean[i], count[i])
    return mean


# # Applying K-mean algorithm

# In[5]:



def k_mean_algo(k,assign,mean):
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
        #print(error)
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
        assign=reassign
        mean=np.zeros([k,2])
        count=np.zeros([k])  
        for i in range(1000):
            mean[int(assign[i])]=mean[int(assign[i])]+array[i]
            count[int(assign[i])]+=1

        for i in range(len(mean)):
            mean[i]=np.divide(mean[i], count[i])
        

    returnlist=[]
    returnlist.append(itlist)
    returnlist.append(errorlist)
    returnlist.append(assign)
    returnlist.append(mean)
    return returnlist

    


# # Error plot Code

# In[6]:


def error_plot(i,itlist,errorlist):
    if(i==0):
        i="FIRST"
    if(i==1):
        i="SECOND"
    if(i==2):
        i="THIRD"
    if(i==3):
        i="FOURTH"
    if(i==4):
        i="FIFTH"
    itlist = np.array(itlist)
    errorlist=np.array(errorlist)
    plt.figure(1)
    titletemp="ERROR CONVERGENCE DIAGRAM FOR" + " " + i + " " "RANDOM INITIALIZATION "
    plt.title(titletemp)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.plot(itlist,errorlist)
    
    


# # Plotting of error and Clusters

# In[7]:



def plot(i,mean,assign,k):
    if(i==0):
        i="FIRST"
    if(i==1):
        i="SECOND"
    if(i==2):
        i="THIRD"
    if(i==3):
        i="FOURTH"
    if(i==4):
        i="FIFTH"
    plt.figure(2)
    titletemp="CLUSTERS FOR" + " " + i + " " "RANDOM INITIALIZATION "
    plt.title(titletemp)
    colour=['red','green','blue','orange','yellow']
    for i in range(1000):
        plt.scatter(array[i][0], array[i][1], c = colour[int(assign[i])])
    for i in range(k):
        plt.scatter(mean[i][0], mean[i][1],c=colour[i],marker='*',s=500)
    plt.show()


# # Starting of the code

# In[ ]:


def starting(i):
    k=4
    assign=initial_assignment(k)
    mean=find_initial_mean(k,assign)
    returnlist=k_mean_algo(k,assign,mean)

    itlist=returnlist[0]
    errorlist=returnlist[1]
    assign=returnlist[2]
    mean=returnlist[3]
    error_plot(i,itlist,errorlist)
    plot(i,mean,assign,k)
for i in range(5):
    starting(i)

