#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# read dataset
df_ekta = pd.read_csv("C:\\Users\\Nisha Jewellers\\Desktop\\rani\\internship project\\Advertising.csv")


# In[3]:


# data analyses
df_ekta.isnull().sum()


# In[4]:


df_ekta['Radio'].mean()


# In[5]:


df_ekta.head(10)


# In[6]:


sns.set()


# In[6]:


df_ekta.describe()


# In[7]:


plt.figure(figsize=(6,6))
sns.distplot(df_ekta['Sales'])
plt.show()


# In[8]:


plt.figure(figsize=(6,6))
sns.distplot(df_ekta['Radio'])
plt.show()


# In[9]:


plt.figure(figsize=(6,6))
sns.distplot(df_ekta['Sales'])
plt.show('Newspaper')


# In[10]:


plt.figure(figsize=(6,6))
sns.distplot(df_ekta['TV'])
plt.show()


# In[11]:


plt.figure(figsize=(6,6))
sns.countplot(x='TV',data=df_ekta)
plt.show()


# In[12]:


plt.figure(figsize=(6,6))
sns.countplot(x='Newspaper',data=df_ekta)
plt.show()


# In[13]:


plt.figure(figsize=(6,6))
sns.countplot(x='Radio',data=df_ekta)
plt.show()


# In[15]:


plt.figure(figsize=(6,6))
sns.countplot(x='Sales',data=df)
plt.show()


# In[14]:


plt.figure(figsize=(30,6))
sns.countplot(x='TV',data=df_ekta)
plt.title('TV')
plt.show()


# In[15]:


sns.pairplot(df_ekta,hue="Sales")


# In[16]:


df_ekta.corr()


# In[17]:


df_ekta.info()


# In[18]:


# classification
df_ekta['Sales'].value_counts()


# In[19]:


# label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[20]:


df_ekta['Sales']=le.fit_transform(df_ekta['Sales'])
df_ekta['Sales'].value_counts()


# In[21]:


df_ekta.head(2)


# In[22]:


x=df_ekta.iloc[:,1:4]
y=df_ekta['Sales']
x.shape,y.shape


# In[23]:


x.head(4)


# In[24]:


y.head(3)


# In[25]:


# train_test_split the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=.25)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[26]:


x_train.head(2)


# In[27]:


x_test.head(2)


# In[28]:


y_train.head(2)


# In[29]:


y_test.head(2)


# In[30]:


# import model/algorithm
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[31]:


# train model
lr.fit(x_train,y_train)


# In[32]:


# predict model
y_pred_lr=lr.predict(x_test)
y_pred_lr[:2]=y_test.values[:2]


# In[33]:


# evaluation
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))


# In[37]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_pred_lr,y_test))
print(classification_report(y_pred_lr,y_test))
print(f' model score-{lr.score(x_test,y_test)} ')
print(f' accuracy score-{accuracy_score(y_pred_lr,y_test)} ')


# In[38]:


# visualization
cm=confusion_matrix(y_pred_lr,y_test)
sns.heatmap(cm,annot=True)
plt.show()

