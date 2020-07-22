#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


data= pd.read_csv("E:/diabetes.csv")


# In[4]:


data.shape


# In[5]:


data.head(5)


# In[6]:


#correlation


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[13]:


#get correlations of each features in dataset
corrmat=data.corr()
data.corr()


# In[11]:


top_corr_features=corrmat.index
plt.figure(figsize=(20,20))


# In[17]:


#plot heat map
g= sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlBu")


# In[19]:


data.head(5)


# In[22]:


outcome_one_count=len(data.loc[data['Outcome']==1])
outcome_zero_count=len(data.loc[data['Outcome']==0])


# In[23]:


(outcome_one_count,outcome_zero_count)


# In[24]:


#Train test split


# In[25]:


from sklearn.model_selection import train_test_split


# In[30]:


#features_columns=['Pregnancies','Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','SkinThickness']
#predicted_class=['Outcome']


# In[31]:


#x= data[features_columns].values
#y= data[predicted_class].values


# In[66]:


train,test=train_test_split(data, test_size=0.2)
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[67]:


train.shape


# In[71]:


train.head(5)


# In[65]:


test.shape


# In[37]:


print("total number of raws:(0)", format(len(data)))


# In[39]:


print("number of rows missing Glucose:(0)", format(len(data.loc[data['Glucose']==0])))


# In[40]:


print("number of rows missing Glucose:(0)", format(len(data.loc[data['Pregnancies']==0])))


# In[41]:


print("number of rows missing Glucose:(0)", format(len(data.loc[data['BloodPressure']==0])))


# In[42]:


print("number of rows missing Glucose:(0)", format(len(data.loc[data['Insulin']==0])))


# In[43]:


print("number of rows missing Glucose:(0)", format(len(data.loc[data['BMI']==0])))


# In[44]:


print("number of rows missing Glucose:(0)", format(len(data.loc[data['DiabetesPedigreeFunction']==0])))


# In[45]:


print("number of rows missing Glucose:(0)", format(len(data.loc[data['Age']==0])))


# In[47]:


print("number of rows missing Glucose:(0)", format(len(data.loc[data['SkinThickness']==0])))


# In[57]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


# In[58]:


#Creat Classifiers
lr=LogisticRegression()


# In[60]:


#GaussianNB()
svc=LinearSVC(C=1.0)
rfc= RandomForestClassifier(n_estimators=100)


# In[72]:


train.iloc[:,:8].head(3)


# In[81]:


train_feat=train.iloc[:,:8]
train_targ=train["Outcome"]


# In[77]:


train_feat.shape


# In[78]:


train_targ.shape


# In[79]:


type(train_targ)


# In[82]:


train[["Outcome"]].info()


# In[ ]:


#logistic regression


# In[83]:


lr.fit(train_feat,train_targ)


# In[ ]:





# In[84]:


lr.score(train_feat,train_targ)


# In[86]:


lr.coef_


# In[108]:


np.transpose(lr.coef_)


# In[113]:


test_feat=test.iloc[:,:8]
test_targ=test["Outcome"]


# In[114]:


test_feat.head(5)


# In[115]:


lr.score(test_feat,test_targ)


# In[116]:


lr.predict(test_feat)


# In[187]:


preds=lr.predict(test_feat)


# In[188]:


pd.crosstab(preds,test_targ)


# In[121]:


from sklearn.metrics import confusion_matrix
confusion_matrix(lr.predict(test_feat),test_targ)


# In[122]:


confusion_matrix(lr.predict(train_feat),train_targ)


# In[ ]:





# In[123]:


#Random forest classifier


# In[124]:


rfc.fit(train_feat,train_targ)


# In[125]:


rfc.score(train_feat,train_targ)


# In[126]:


rfc.score(test_feat,test_targ)


# In[127]:


rfc.predict(train_feat)[:10]


# In[128]:


confusion_matrix(rfc.predict(train_feat),train_targ)


# In[129]:


confusion_matrix(rfc.predict(test_feat),test_targ)


# In[181]:


pr_y=rfc.predict(test_feat)


# In[182]:


#model accuracy


# In[183]:


from sklearn import metrics


# In[184]:


Ac=metrics.accuracy_score(test_targ,pr_y)
print(Ac)


# In[ ]:





# In[130]:


#SVM (linear Support vector machine classifier)


# In[131]:


svc.fit(train_feat,train_targ)


# In[132]:


svc.score(train_feat,train_targ)


# In[133]:


svc.score(test_feat,test_targ)


# In[134]:


svc.coef_


# In[135]:


np.transpose(svc.coef_)


# In[179]:


ypred=svc.predict(test_feat)


# In[178]:


#model accuracy


# In[ ]:


from sklearn import metrics


# In[180]:


Ac=metrics.accuracy_score(test_targ,ypred)
print(Ac)


# In[136]:


svc.predict(train_feat)


# In[137]:


preds=svc.predict(train_feat)


# In[138]:


pd.crosstab(preds,train_targ)


# In[139]:


np.sum(np.diag(pd.crosstab(preds,train_targ)))


# In[143]:


from sklearn.metrics import confusion_matrix


# In[147]:


confusion_matrix(svc.predict(train_feat),train_targ)


# In[148]:


confusion_matrix(svc.predict(test_feat),test_targ)


# In[150]:


#Decision tree classifier


# In[153]:


from sklearn.tree import DecisionTreeClassifier


# In[170]:


clf=DecisionTreeClassifier(criterion="entropy", max_depth=3)


# In[163]:


#train decision tree


# In[171]:


clf.fit(train_feat,train_targ)


# In[172]:


y_pred=clf.predict(test_feat)


# In[ ]:


# model accuracy


# In[174]:


from sklearn import metrics


# In[176]:


Acc=metrics.accuracy_score(test_targ,y_pred)
print(Acc)


# In[177]:


y_pred[0:]


# In[ ]:




