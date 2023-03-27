#!/usr/bin/env python
# coding: utf-8

# In[15]:


# import the module
import numpy as np
import pandas as pd
import wordcloud


# In[16]:


import os
# print(os.listdir("C:\Users\Nisha Jewellers\Desktop\rani\internship project\spam"))


# In[17]:


# read the dataset
data_ekta=pd.read_csv("C:\\Users\\Nisha Jewellers\\Desktop\\rani\\internship project\\spam.csv",encoding="latin-1")


# In[18]:


# analyses
data_ekta.shape
data_ekta.head()


# In[19]:


# dropping the unused columns
data_ekta=data_ekta.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)


# In[20]:


# rename the columns
data_ekta=data_ekta.rename(columns={'v1':'Type','v2':'Messages'})
data_ekta.columns


# In[22]:


# filter the spam messages
df=pd.DataFrame(data_ekta)
Spamfilter=df.loc[df['Type']=='spam']
print(Spamfilter)


# In[23]:


# filter the ham messages
df=pd.DataFrame(data_ekta)
hamfilter=df.loc[df['Type']=='ham']
print(hamfilter)


# In[24]:


# print the most common number of words used in messages
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud=WordCloud(background_color='White',width=1000,height=1000,max_words=50).generate(str(data_ekta['Messages']))
plt.rcParams['figure.figsize']=(10,10)
plt.title('Most common number of words in Messages')
plt.axis('off')
plt.imshow(wordcloud)


# In[25]:


# print the most used words in spam messages
wordcloud=WordCloud(background_color='White',width=1000,height=1000,max_words=50).generate(str([Spamfilter]))
plt.rcParams['figure.figsize']=(10,10)
plt.title('Most common number of words in Spam Messages')
plt.axis('off')
plt.imshow(wordcloud)


# In[26]:


# # print the most used words in ham messages
wordcloud=WordCloud(background_color='White',width=1000,height=1000,max_words=50).generate(str([hamfilter]))
plt.rcParams['figure.figsize']=(10,10)
plt.title('Most common number of words in ham Messages')
plt.axis('off')
plt.imshow(wordcloud)

