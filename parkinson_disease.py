#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip3 install xgboost ')


# In[14]:


#Necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[6]:


#Reading the data
df=pd.read_csv("parkinsons.csv")
df.head()


# In[7]:


#  Using .info() method to examine each column data type and possible missing data
df.info()


# In[8]:


#Using .describe() method to see data summary statictic, such as min, median, max, and so on
df.describe()


# In[9]:


#Extracting the features and labels
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values


# In[11]:


#Correlation between features
df.corr()


# In[17]:


# Plot heatmap
fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(df.corr(), annot=True);


# In[18]:


# Plot using pie chart
df.status.value_counts().plot.pie()


# In[19]:


#Getting the count of each label (0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])


# In[20]:


#Scaling the features between -1 and 1 to normalize them
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels


# In[21]:


#Splitting the dataset into training and testing sets keeping 20% data for testing
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


# In[22]:


#Initializing an XGB classifier and training the model
model=XGBClassifier()
model.fit(x_train,y_train)


# In[23]:


#Generating y_pred and calculating the accuracy for the model
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)


# In[ ]:




