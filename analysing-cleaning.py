#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd                      
import numpy as np                       
import seaborn as sns                   
import matplotlib.pyplot as plt          
get_ipython().run_line_magic('matplotlib', 'inline')
import math

diabet_data= pd.read_csv("E:/diabetes.csv")
diabet_data.head(10)


# In[3]:


print("number of tested people original data:" +str((len(diabet_data))))


# ##Analysing data

# In[4]:


sns.countplot(x="Pregnancies",data=diabet_data)


# In[5]:


sns.countplot(x="Glucose",data=diabet_data)


# In[6]:


sns.countplot(x="BloodPressure",data=diabet_data)


# In[7]:


sns.countplot(x="SkinThickness",data=diabet_data)


# In[8]:


sns.countplot(x="Insulin",data=diabet_data)


# In[9]:


sns.countplot(x="BMI",data=diabet_data)


# In[11]:


sns.countplot(x="DiabetesPedigreeFunction",data=diabet_data)


# In[12]:


sns.countplot(x="Age",data=diabet_data)


# In[13]:


sns.countplot(x="Outcome",data=diabet_data)


# ##Number of times pregnant
# 
# 

# In[14]:


diabet_data["Pregnancies"].plot.hist()


# ##Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# 

# In[15]:


diabet_data["Glucose"].plot.hist()


# ##Diastolic blood pressure (mm Hg)
# 
# 

# In[16]:


diabet_data["BloodPressure"].plot.hist()


# ##Triceps skin fold thickness (mm)
# 
# 

# In[17]:


diabet_data["SkinThickness"].plot.hist()


# ##2-Hour serum insulin (mu U/ml)

# In[18]:


diabet_data["Insulin"].plot.hist()


# ##Body mass index (weight in kg/(height in m)^2)

# In[19]:


diabet_data["BMI"].plot.hist()


# ##Diabetes pedigree function

# In[20]:


diabet_data["DiabetesPedigreeFunction"].plot.hist()


# ##Age

# In[21]:


diabet_data["Age"].plot.hist()


# ##Class variable (0 or 1) 268 of 768 are 1, the others are 0

# In[22]:


diabet_data["Outcome"].plot.hist()


# In[23]:


diabet_data.info()


# ##Data Wrangling

# In[24]:


diabet_data.isnull()


# In[25]:


diabet_data.isnull().sum()


# In[26]:


sns.heatmap(diabet_data.isnull(),"yticklabeles" ==False)


# In[27]:


diabet_data.head()


# ###Histogram and density graphs of all variables

# In[29]:


fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(diabet_data.Age, bins = 20, ax=ax[0,0]) 
sns.distplot(diabet_data.Pregnancies, bins = 20, ax=ax[0,1]) 
sns.distplot(diabet_data.Glucose, bins = 20, ax=ax[1,0]) 
sns.distplot(diabet_data.BloodPressure, bins = 20, ax=ax[1,1]) 
sns.distplot(diabet_data.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(diabet_data.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(diabet_data.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sns.distplot(diabet_data.BMI, bins = 20, ax=ax[3,1])


# In[30]:


my_colors = ['lightblue','lightsteelblue','silver']
ax = diabet_data["Outcome"]
ax.value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True,colors=my_colors)
plt.show()


# In[31]:


f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(diabet_data.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


# In[70]:


print(diabet_data.groupby("Outcome").agg({"Pregnancies":"mean"}))
print(diabet_data.groupby("Outcome").agg({"Age":"mean"}))
print(diabet_data.groupby("Outcome").agg({"Insulin": "mean"}))
print(diabet_data.groupby("Outcome").agg({"Glucose": "mean"}))
print(diabet_data.groupby("Outcome").agg({"BMI": "mean"}))

