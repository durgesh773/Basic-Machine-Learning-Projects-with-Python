#!/usr/bin/env python
# coding: utf-8

# # Stock Sentiment Analysis Using News Headlines

# In[1]:


import pandas as pd 


# In[3]:


df =  pd.read_csv('Data.csv', encoding = "ISO-8859-1")


# In[4]:


df.head()


# In[5]:


train  = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']


# In[6]:


#removing punctuation 
data = train.iloc[:,2:27]
data.replace("^a-zA-Z"," ",regex=True, inplace = True)


# In[7]:


#Renaming column names for ease of access
list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index
data.head(5)


# In[8]:


#converting headlines to lower case
for index in new_Index:
    data[index] = data[index].str.lower()
data.head(1)


# In[10]:


' '.join(str(x) for x in data.iloc[1,0:25])


# In[11]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[12]:


headlines[0]


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[14]:


#implement Bag Of Words
countvector = CountVectorizer(ngram_range=(2,2))
traindataset = countvector.fit_transform(headlines)


# In[15]:


#implement RandomForest Classifier
randomclassifier = RandomForestClassifier(n_estimators=200,criterion ='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[16]:


#Predict for the test dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)


# In[17]:


## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[18]:


matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)

