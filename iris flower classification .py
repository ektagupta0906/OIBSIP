#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# read dataset
columns =['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species']
df_ekta = pd.read_csv("C:\\Users\\Nisha Jewellers\\Desktop\\rani\\internship project\\Iris.csv")
df1=df_ekta


# In[3]:


# data analyses
df_ekta.head(10)


# In[4]:


df_ekta.describe()


# In[5]:


df_ekta.info()


# In[6]:


sns.set_style("whitegrid")
sns.pairplot(df_ekta,hue="Species");
plt.show()


# In[8]:


# classification
# logistic regression
df_ekta['Species'].value_counts()


# In[9]:


df_ekta.head(2)


# In[10]:


x=df_ekta.iloc[:,1:5]
y=df_ekta['Species']
x.shape,y.shape


# In[11]:


x.head(3)


# In[12]:


y.head(3)


# In[13]:


# train_test_split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.25)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
# print(x_test)


# In[14]:


x_train.head(2)


# In[15]:


x_test.head(2)


# In[16]:


y_train.head(2)


# In[17]:


y_test.head(2)


# In[18]:


# import model/algorithm
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[19]:


# train model
lr.fit(x_train,y_train)


# In[20]:


# predict model
y_pred_lr=lr.predict(x_test)
y_pred_lr[:5],y_test.values[:5]


# In[21]:


# evaluation
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))


# In[22]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_pred_lr,y_test))
print(classification_report(y_pred_lr,y_test))
print(f'model_score- {lr.score(x_test,y_test)} ')
print(f'accuracy_score- {accuracy_score(y_pred_lr,y_test)} ')


# In[23]:


# visualization
cm=confusion_matrix(y_pred_lr,y_test)
sns.heatmap(cm,annot=True)
plt.show()


# In[24]:


# decision tree classifier
# import the model/algorithm
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()


# In[25]:


# train model
dtc.fit(x_train,y_train)


# In[30]:


# predict model, evaluation and visualization
y_pred_dtc=dtc.predict(x_test)
print(f'Predicted_y{y_pred_dtc[:5]} Actual_y{y_test.values[:5]}')
print(confusion_matrix(y_pred_dtc,y_test))
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_pred_dtc,y_test)
plt.figure(figsize=(2,1))
print(sns.heatmap(cm,annot=True))
plt.show()
print(classification_report(y_pred_dtc,y_test))
print(f'model_score- {dtc.score(x_test,y_test)} ')
print(f'accuracy_score- {accuracy_score(y_pred_dtc,y_test)} ')


# In[31]:


# random forest classifier
# import the model
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()


# In[32]:


# train model
rfc.fit(x_train,y_train)


# In[33]:


# predict model, evaluation and visualization 
y_pred_rfc=rfc.predict(x_test)
print(f'predicted_y-{y_pred_rfc} actual_y-{y_test.values}')
print(confusion_matrix(y_pred_rfc,y_test))
# from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_pred_rfc,y_test)
plt.figure(figsize=(2,1))
sns.heatmap(cm,annot=True)
plt.show()
print(classification_report(y_pred_rfc,y_test))
print(f'model_score- {rfc.score(x_test,y_test)} ')
print(f'accuracy_score- {accuracy_score(y_pred_rfc,y_test)} ')

