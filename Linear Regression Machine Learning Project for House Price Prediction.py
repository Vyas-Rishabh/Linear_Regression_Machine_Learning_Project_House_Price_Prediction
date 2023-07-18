#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Machine Learning Project for House Price Prediction

# importing Seaborn, Pandas, Seaborn, Matplotlib and Numpy.

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Data and Checking out

# In[4]:


houseDf = pd.read_csv(r"C:\Users\RISHABH\Documents\discount calculating\USA_Housing.csv")


# In[5]:


houseDf.head()


# In[6]:


houseDf.info()


# In[7]:


houseDf.describe()


# In[8]:


houseDf.columns


# # Exploratory Data Analysis for House Price Prediction

# In[9]:


sns.pairplot(houseDf)


# In[10]:


sns.heatmap(houseDf.corr(), annot=True)


# # Get Data Ready For Training a Linear Regression Model

# # X and y List

# In[11]:


X = houseDf[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = houseDf['Price']


# In[12]:


from sklearn.model_selection import train_test_split


# # Split Data into Train, Test

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)


# # Creating and Training the LinearRegression Model

# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


lm = LinearRegression()


# In[16]:


lm.fit(X_train,y_train)


# # LinearRegression Model Evaluation

# In[17]:


print(lm.intercept_)


# In[18]:


coeff_df = pd.DataFrame(lm.coef_, X.columns,columns=['Coefficient'])


# In[19]:


coeff_df


# # Predictions from our Linear Regression Model

# In[20]:


predictions = lm.predict(X_test)


# In[21]:


plt.scatter(y_test, predictions)


# In the above scatter plot, we see data is in a line form, which means our model has done good predictions.

# In[22]:


sns.distplot((y_test-predictions), bins=50);


# In the above histogram plot, we see data is in bell shape (Normally Distributed), which means our model has done good predictions.

# # Regression Evaluation Metrics

# In[23]:


from sklearn import metrics


# In[26]:


print("MAE:",metrics.mean_absolute_error(y_test, predictions))
print("MSE:",metrics.mean_squared_error(y_test, predictions))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# # Conclusion
# 

# We have created a Linear Regression Model which we help the real state agent for estimating the house price.
# 
# You can find this project on <a href="https://github.com/Vyas-Rishabh/Linear_Regression_Machine_Learning_Project_House_Price_Prediction"><B>GitHub</B></a>.
