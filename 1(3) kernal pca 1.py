#!/usr/bin/env python
# coding: utf-8

# # Answer 1.(3)(a)
# # Kernal PCA with d=2

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\amitt\OneDrive\Desktop\Dataset.csv",header=None)
df.columns=('x','y')
array=df.to_numpy()
print(array)


# In[3]:


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


# In[4]:


eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()
eigenval1=eig_vals[0]
eigenval2=eig_vals[1]
eigenvec1=eig_vecs[0]
eigenvec2=eig_vecs[1]


# In[5]:


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


# In[6]:


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


# In[7]:


plt.scatter(c1,c2)


# In[ ]:











# # Aswer
# # kernal PCA with d=3

# In[8]:


array=df.to_numpy()


# In[9]:


transposearray=np.transpose(array)
k=np.matmul(array, transposearray)
for i in range(1000):
    for j in range(1000):
        k[i][j]+=1 
for i in range(1000):
    for j in range(1000):
        k[i][j]=k[i][j]*k[i][j]*k[i][j]
I=np.identity(1000) 
div=1/1000
divmat=np.full((1000,1000), div)
matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()


# In[10]:


maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


# In[11]:


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


# In[12]:


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


# In[13]:


plt.scatter(c1,c2)


# # Answer 1.(3)(B)
# 

# # sigma=.1

# In[14]:


array=df.to_numpy()


# In[15]:


tempmat=np.zeros([1000,1000])


# In[16]:


sigma=0.1
div = sigma**2
div *= -2

for i in range(1000):
    for j in range(1000):
        x = array[i]
        y = array[j]
        sub = np.subtract(x,y)
        var = np.matmul(sub, sub.transpose())
        var = var / (2*sigma*sigma)
        var = np.exp(-1*var)
        tempmat[i][j] = var


# In[17]:


x = array[2]
y = array[3]
sub = np.subtract(x,y)
var = np.matmul(sub, sub.transpose())
var = var / (2*sigma*sigma)
var = np.exp(-1*var)


# In[18]:


k = tempmat
I=np.identity(1000)
div=1/1000
divmat=np.full((1000,1000), div)
divmat


# In[19]:


matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()


# In[20]:


maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


# In[21]:


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


# In[22]:


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


# In[23]:


plt.scatter(c1,c2)


# In[ ]:





# # # sigma=.2

# In[24]:


array=df.to_numpy()
tempmat=np.zeros([1000,1000])
sigma=0.2
div = sigma**2
div *= -2

for i in range(1000):
    for j in range(1000):
        x = array[i]
        y = array[j]
        sub = np.subtract(x,y)
        var = np.matmul(sub, sub.transpose())
        var = var / (2*sigma*sigma)
        var = np.exp(-1*var)
        tempmat[i][j] = var
x = array[2]
y = array[3]
sub = np.subtract(x,y)
var = np.matmul(sub, sub.transpose())
var = var / (2*sigma*sigma)
var = np.exp(-1*var)

k = tempmat
I=np.identity(1000)
div=1/1000
divmat=np.full((1000,1000), div)
divmat


matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()

maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


plt.scatter(c1,c2)


# # sigma=.3

# In[25]:


array=df.to_numpy()
tempmat=np.zeros([1000,1000])
sigma=0.3
div = sigma**2
div *= -2

for i in range(1000):
    for j in range(1000):
        x = array[i]
        y = array[j]
        sub = np.subtract(x,y)
        var = np.matmul(sub, sub.transpose())
        var = var / (2*sigma*sigma)
        var = np.exp(-1*var)
        tempmat[i][j] = var
x = array[2]
y = array[3]
sub = np.subtract(x,y)
var = np.matmul(sub, sub.transpose())
var = var / (2*sigma*sigma)
var = np.exp(-1*var)

k = tempmat
I=np.identity(1000)
div=1/1000
divmat=np.full((1000,1000), div)
divmat


matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()

maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


plt.scatter(c1,c2)


# #  sigma=.4

# In[26]:


array=df.to_numpy()
tempmat=np.zeros([1000,1000])
sigma=0.4
div = sigma**2
div *= -2

for i in range(1000):
    for j in range(1000):
        x = array[i]
        y = array[j]
        sub = np.subtract(x,y)
        var = np.matmul(sub, sub.transpose())
        var = var / (2*sigma*sigma)
        var = np.exp(-1*var)
        tempmat[i][j] = var
x = array[2]
y = array[3]
sub = np.subtract(x,y)
var = np.matmul(sub, sub.transpose())
var = var / (2*sigma*sigma)
var = np.exp(-1*var)

k = tempmat
I=np.identity(1000)
div=1/1000
divmat=np.full((1000,1000), div)
divmat


matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()

maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


plt.scatter(c1,c2)


# # sigma=.5

# In[27]:


array=df.to_numpy()
tempmat=np.zeros([1000,1000])
sigma=0.5
div = sigma**2
div *= -2

for i in range(1000):
    for j in range(1000):
        x = array[i]
        y = array[j]
        sub = np.subtract(x,y)
        var = np.matmul(sub, sub.transpose())
        var = var / (2*sigma*sigma)
        var = np.exp(-1*var)
        tempmat[i][j] = var
x = array[2]
y = array[3]
sub = np.subtract(x,y)
var = np.matmul(sub, sub.transpose())
var = var / (2*sigma*sigma)
var = np.exp(-1*var)

k = tempmat
I=np.identity(1000)
div=1/1000
divmat=np.full((1000,1000), div)
divmat


matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()

maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


plt.scatter(c1,c2)


# # sigma=.6

# In[28]:


array=df.to_numpy()
tempmat=np.zeros([1000,1000])
sigma=0.6
div = sigma**2
div *= -2

for i in range(1000):
    for j in range(1000):
        x = array[i]
        y = array[j]
        sub = np.subtract(x,y)
        var = np.matmul(sub, sub.transpose())
        var = var / (2*sigma*sigma)
        var = np.exp(-1*var)
        tempmat[i][j] = var
x = array[2]
y = array[3]
sub = np.subtract(x,y)
var = np.matmul(sub, sub.transpose())
var = var / (2*sigma*sigma)
var = np.exp(-1*var)

k = tempmat
I=np.identity(1000)
div=1/1000
divmat=np.full((1000,1000), div)
divmat


matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()

maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


plt.scatter(c1,c2)


# # sigma=.7

# In[29]:


array=df.to_numpy()
tempmat=np.zeros([1000,1000])
sigma=0.7
div = sigma**2
div *= -2

for i in range(1000):
    for j in range(1000):
        x = array[i]
        y = array[j]
        sub = np.subtract(x,y)
        var = np.matmul(sub, sub.transpose())
        var = var / (2*sigma*sigma)
        var = np.exp(-1*var)
        tempmat[i][j] = var
x = array[2]
y = array[3]
sub = np.subtract(x,y)
var = np.matmul(sub, sub.transpose())
var = var / (2*sigma*sigma)
var = np.exp(-1*var)

k = tempmat
I=np.identity(1000)
div=1/1000
divmat=np.full((1000,1000), div)
divmat


matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()

maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


plt.scatter(c1,c2)


# # sigma=.8

# In[30]:


array=df.to_numpy()
tempmat=np.zeros([1000,1000])
sigma=0.8
div = sigma**2
div *= -2

for i in range(1000):
    for j in range(1000):
        x = array[i]
        y = array[j]
        sub = np.subtract(x,y)
        var = np.matmul(sub, sub.transpose())
        var = var / (2*sigma*sigma)
        var = np.exp(-1*var)
        tempmat[i][j] = var
x = array[2]
y = array[3]
sub = np.subtract(x,y)
var = np.matmul(sub, sub.transpose())
var = var / (2*sigma*sigma)
var = np.exp(-1*var)

k = tempmat
I=np.identity(1000)
div=1/1000
divmat=np.full((1000,1000), div)
divmat


matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()

maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


plt.scatter(c1,c2)


# # sigma=.9

# In[31]:


array=df.to_numpy()
tempmat=np.zeros([1000,1000])
sigma=0.9
div = sigma**2
div *= -2

for i in range(1000):
    for j in range(1000):
        x = array[i]
        y = array[j]
        sub = np.subtract(x,y)
        var = np.matmul(sub, sub.transpose())
        var = var / (2*sigma*sigma)
        var = np.exp(-1*var)
        tempmat[i][j] = var
x = array[2]
y = array[3]
sub = np.subtract(x,y)
var = np.matmul(sub, sub.transpose())
var = var / (2*sigma*sigma)
var = np.exp(-1*var)

k = tempmat
I=np.identity(1000)
div=1/1000
divmat=np.full((1000,1000), div)
divmat


matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()

maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


plt.scatter(c1,c2)


# # sigma=1

# In[32]:


array=df.to_numpy()
tempmat=np.zeros([1000,1000])
sigma=1
div = sigma**2
div *= -2

for i in range(1000):
    for j in range(1000):
        x = array[i]
        y = array[j]
        sub = np.subtract(x,y)
        var = np.matmul(sub, sub.transpose())
        var = var / (2*sigma*sigma)
        var = np.exp(-1*var)
        tempmat[i][j] = var
x = array[2]
y = array[3]
sub = np.subtract(x,y)
var = np.matmul(sub, sub.transpose())
var = var / (2*sigma*sigma)
var = np.exp(-1*var)

k = tempmat
I=np.identity(1000)
div=1/1000
divmat=np.full((1000,1000), div)
divmat


matminus = np.subtract(I, divmat) 
f=np.matmul(matminus, k)
kernalcent=np.matmul(f, matminus)
eig_vals, eig_vecs = np.linalg.eig(kernalcent)
eig_vecs = eig_vecs.transpose()

maxindex=np.argmax(eig_vals)
eigenval1=eig_vals[maxindex]
eigenvec1=eig_vecs[maxindex]
eig_vals=np.delete(eig_vals,maxindex,axis=0)
eig_vecs=np.delete(eig_vecs,maxindex,axis=0)
maxindex=np.argmax(eig_vals)
eigenval2=eig_vals[maxindex]
eigenvec2=eig_vecs[maxindex]


import math 
e1=math.sqrt(eigenval1)
e2=math.sqrt(eigenval2)
alpha1=np.divide(eigenvec1, e1)
alpha2=np.divide(eigenvec2, e2)


c1=np.matmul(kernalcent, alpha1)
c2=np.matmul(kernalcent, alpha2)


plt.scatter(c1,c2)


# In[ ]:





# In[ ]:




