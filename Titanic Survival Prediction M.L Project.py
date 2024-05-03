#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Data Prediction

# ### Steps in this project

# * Data Collection
# * Data Preprocessing
# * Data Analysis
# * Split data into Train and Test
# * Logistic Regression Model Building
# * Evaluation

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ## Data Collection and Preprocessing

# In[2]:


df=pd.read_csv("C:/Users/ADMIN/Downloads/train.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


# no.of rows and columns in data
df.shape


# In[6]:


# information of data
df.info()


# In[7]:


# no of missing values in data
df.isnull().sum()


# ### Handling the missing values

# In[8]:


# Drop 'Cabin' column from dataframe
df=df.drop(columns='Cabin',axis=1)


# In[9]:


# filling the missing Age values by mean Age
mean=df['Age'].mean()


# In[10]:


mean


# In[11]:


df['Age']=df['Age'].fillna(mean)


# In[12]:


#for Embarked column we are finding most repetet value(mode)
mode=df['Embarked'].mode()


# In[13]:


mode


# In[14]:


#replacling missing values in Embarked column with mode value
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)


# In[15]:


# There is no anynull value in the data
df.isnull().sum()


# # Data Analysis

# In[16]:


# getting some statistical measures about the data
df.describe().T


# In[17]:


# finding the number of people survived and not survived
df['Survived'].value_counts()


# In[18]:


# survival rate by sex
df.groupby('Sex')[['Survived']].mean()*100


# In[19]:


# survival rate by sex and class
df.pivot_table('Survived',index='Sex',columns='Pclass')


# In[20]:


# survival rate by sex and class visually
df.pivot_table('Survived',index='Sex',columns='Pclass').plot()


# In[21]:


sns.barplot(x='Pclass',y='Survived',data=df)


# In[22]:


# survival rate by sex,age and class
age=pd.cut(['Age'],[0,18,80])
df.pivot_table('Survived',['Sex','Age'],'Pclass')


# In[23]:


# finding the number of count on the basis of sex
df['Sex'].value_counts()


# In[24]:


# finding percentage of survival(1) and non survival(0)
plt.pie(df['Survived'].value_counts(),labels=['0','1'],autopct='%.2f')


# In[25]:


# count on the basis of 'sex'
sns.countplot(x='Sex',data=df)


# In[26]:


# number of survivors Gender wise
sns.countplot(x='Sex',hue='Survived',data=df)


# 
# 
# 
# * import insight we get from the data
# '''getting from above is even though we have more number of males in our data set,the number of female who have survived 
# in the titanic accident is more'''

# In[27]:


# count on the basis of 'Pclass '
sns.countplot(x='Pclass',data=df)


# In[28]:


sns.countplot(x='Pclass',hue='Survived',data=df)


# # Encoding
# convert categorical column into numerical

# In[29]:


df['Sex'].value_counts()


# In[30]:


df['Embarked'].value_counts()


# In[31]:


# converting categorical columns
df.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[32]:


df.head()


# In[33]:


print(df['Embarked'].unique())
print(df['Sex'].unique())


# In[34]:


# look at all values in each column and get a count
for val in df:
    print(df[val].value_counts())
    print()


# # Separating Features and Target

# In[35]:


x=df.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
y=df['Survived']


# In[36]:


print(x)


# In[37]:


print(y)


# # Splitting the data into Training data and Test data

# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import accuracy_score


# In[39]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[40]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[41]:


# Scale the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# # Model training

# ### Logistic Regression

# In[42]:


lr=LogisticRegression()


# In[43]:


lr.fit(x_train,y_train)


# # Model Evaluation

# ## Accuracy score

# In[52]:


# accuracy on training data
x_train_prediction=lr.predict(x_train)
print(x_train_prediction)


# In[ ]:


# we are compring predicted values(x_train) with original values(y_train)


# In[55]:


training_data_accuracy=accuracy_score(y_train,x_train_prediction)
print("Accuracy score of training data: ",training_data_accuracy*100)


# In[56]:


# accuracy on test data
x_test_prediction=lr.predict(x_test)
print(x_test_prediction)


# In[57]:


test_data_accuracy=accuracy_score(y_test,x_test_prediction)
print("Accuracy score of test data: ",test_data_accuracy*100)


#  * Achieving a high level of accuracy in predicting survival outcomes

# # The End
