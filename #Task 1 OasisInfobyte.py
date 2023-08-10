#!/usr/bin/env python
# coding: utf-8

# # Task1 : Iris Flower Classification.

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[5]:


df= pd.read_csv("C:\\Users\\trupti kadam\\Downloads\\Oasisinfobyte.csv")
df


# In[6]:


df.head(20)


# In[7]:


df.info()


# In[8]:


# checking if there are any null values
df.isnull().sum()


# In[9]:


# view the columns present in the dataset
df.columns


# In[10]:


df.describe()


# In[11]:


# Deleting unnecessary column
iris=df.drop('Id', axis=1)
iris


# # Visualization

# In[12]:


iris['Species'].value_counts()


# In[13]:


sns.countplot(iris['Species'])


# In[14]:


# dividing data into input and output variables
x= iris.iloc[:,0:4]
y= iris.iloc[:,4]


# In[15]:


x


# In[16]:


y


# # Train Test Split

# In[17]:


# splitting the dataset into train and test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,random_state = 0)


# In[18]:


x_train.shape


# In[19]:


x_test.shape


# In[20]:


y_train.shape


# In[21]:


y_test.shape


# # Classification Model

# In[22]:


from sklearn.linear_model import LogisticRegression
model= LogisticRegression()


# In[23]:


# Fitting the Model
model.fit(x_train, y_train)


# In[24]:


# Predicting the result
y_pred= model.predict(x_test)


# In[25]:


y_pred


# # Accuracy of Model

# In[26]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy = accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}". format(accuracy))


# In[27]:


# making a prediction for example
model.predict([[4,3.5,2,0]])


# In[ ]:




