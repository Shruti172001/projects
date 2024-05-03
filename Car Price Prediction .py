#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[27]:


df=pd.read_csv("D:\c folder files\car_price_prediction - car_price_prediction (1).csv")


# In[28]:


df


# In[29]:


df.head()


# In[30]:


df.columns


# In[31]:


df.isnull().sum()


# In[32]:


df1=df.dropna()


# In[33]:


df1.isnull().sum()


# In[34]:


df1["Brand"].value_counts(normalize=True)*100


# In[35]:


plt.pie(df1["Brand"].value_counts(normalize=True)*100,labels=["Volkswagen","Mercedes-Benz","BMW","Toyota","Renault","Audi","Mitsubishi"])


# In[36]:


df1["Engine Type"].value_counts(normalize=True)*100


# In[37]:


plt.pie(df1["Engine Type"].value_counts(normalize=True)*100,labels=list(df1["Engine Type"].uni))


# In[38]:


plt.pie(df1["Registration"].value_counts(normalize=True)*100,labels=list(df1["Registration"].unique()))


# In[39]:


plt.pie(df1["Body"].value_counts(normalize=True)*100,labels=list(df1["Body"].unique()))


# In[40]:


plt.pie(df1["Year"].value_counts(normalize=True)*100,labels=list(df1["Year"].unique()))


# # skewness

# In[41]:


df1.dtypes


# In[42]:


df1["Price"].skew()


# In[43]:


df1["Year"].skew()


# In[44]:


df1["Mileage"].skew()


# In[45]:


df1["EngineV"].skew()


# In[46]:


sns.pairplot(data=df1)


# In[47]:


df1.dtypes


# In[48]:


df1["Price"]=df1["Price"].astype("int")
df1["EngineV"]=df1["EngineV"].astype("int")


# In[49]:


df1.dtypes


# In[50]:


df2=df1.dropna()


# In[51]:


df2.isnull().sum()


# # Encoding

# In[52]:


le=LabelEncoder()


# In[53]:


l1=le.fit_transform(df1["Brand"])
l2=le.fit_transform(df1["Body"])
l3=le.fit_transform(df1["Engine Type"])
l4=le.fit_transform(df1["Registration"])
l5=le.fit_transform(df1["Model"])


# In[54]:


df1["Brand"]=l1
df1["Body"]=l2
df1["Engine Type"]=l3
df1["Registration"]=l4
df1["Model"]=l5


# In[55]:


df1.corr()


# In[56]:


sns.heatmap(df1.corr(),annot=True)


# # model building without scalling

# * 1.segregating data into x and y
# * 2.segregating x into x test , x train,y into y test and y train
# * 3.building model on train data
# * 4.testing model on test data
# 

# * we check score without scaling data

# In[57]:


df.head()


# ## splitting data into x and y

# In[58]:


x=df1.drop(["Price"],axis=1)                   # price is our target


# In[59]:


y=df1["Price"]


# In[60]:


x.head()


# In[61]:


from sklearn.linear_model import LinearRegression              # algorithm
from sklearn.model_selection import train_test_split           # spliting into train and test
from sklearn.metrics import r2_score,mean_squared_error        # 


# In[62]:


y.head()


#  * splitting x into x train,x test

# In[63]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)             


# In[64]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## building the model

# In[65]:


lr=LinearRegression()


# ## training model

# In[66]:


lr.fit(x_train,y_train)


# In[67]:


# prediction is done on x test
# and actual value is y test
# model check the x data to that our prediction about y is true or false


# In[68]:


y_true,y_pred=y_test,lr.predict(x_test)


# In[69]:


y_true.head()


# ## Evaluation of model

# In[70]:


r2_score(y_true,y_pred)            # in regression always use a r2 score 
                                   # more r2 score the model is good (it lies bet 0 to 1)


# In[71]:


mean_squared_error(y_true,y_pred)


# # Ridge and Hasso Regression

# In[72]:


from sklearn.linear_model import Ridge


# In[73]:


re=Ridge()


# In[74]:


re.fit(x_train,y_train)


# In[82]:


y_true,y_pred=y_test,re.predict(x_test)


# In[88]:


r2_score(y_true,y_pred)*100              # accuracy of model is 34% 


# In[87]:


mean_squared_error(y_true,y_pred)


# In[ ]:


from sklearn.linear_model import 

