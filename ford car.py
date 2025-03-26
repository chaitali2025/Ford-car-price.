#!/usr/bin/env python
# coding: utf-8

# ford car dataset:: numerical target data::

# In[ ]:


import pandas as pd
#path=E:\Python Dataset\BankCreditCard.csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings 
warnings.filterwarnings('ignore')
#ford=ford.drop(["price"],axis=1)
from sklearn import metrics
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge
from sklearn import metrics


# In[3]:


ford=pd.read_csv(r"E:\Python Dataset\ford.csv")


# In[4]:


ford.isnull().sum()


# In[5]:


ford.fuelType.value_counts()


# 2a.#no.of sales of each fuel types are ,Petrol=12179,Diesel= 5762,Hybrid=22,Electric=2,Other=1.

# In[7]:


ford.transmission.value_counts()


# In[8]:


ford.model.value_counts()


# In[9]:


ford.dtypes


# In[10]:


le=LabelEncoder()


# In[11]:


ford["model"]=le.fit_transform(ford["model"])


# In[12]:


ford["model"]


# In[13]:


ford["transmission"]=le.fit_transform(ford["transmission"])


# In[14]:


ford["fuelType"]=le.fit_transform(ford["fuelType"])


# In[15]:


ford["transmission"]


# In[16]:


ford["fuelType"]


# In[17]:


ford.model.value_counts()


# In[18]:


ford.dtypes


# In[19]:


ford.shape


# In[20]:


ford.drop_duplicates(inplace=True)


# In[21]:


ford.shape


# In[22]:


#new_ford1=ford[ford['transmission']=='Manual']


# In[23]:


#new_ford1.transmission=new_ford1.replace({"Manual":0},inplace=True)


# In[24]:


new_ford1.tail()


# In[ ]:


#new_ford2=ford[ford['transmission']=='Automatic']


# In[ ]:


new_ford2.head()


# In[ ]:


#new_ford3=ford[ford['transmission']=='Semi-Auto']


# In[ ]:


new_ford3.head()


# In[ ]:


new_ford1=new_ford1[['transmission','price']]


# In[ ]:


import seaborn as sns
sns.boxplot(data=new_ford1,y="price",x="transmission")


# In[ ]:


import seaborn as sns
sns.boxplot(data=new_ford2,y="price",x="transmission")


# In[ ]:


import seaborn as sns
sns.boxplot(data=new_ford3,y="price",x="transmission")


# #2b.By performing sagrigation of 3 types of transmission in different dataframe I can able to boxplot the target  price outlier .
# As aresult MANUAL type transmission type.

# In[ ]:


import seaborn as sns
sns.boxplot(data=ford,y="price",x="transmission")


# train_test_split::

# In[54]:


x=ford.iloc[:,:8]
x.shape
x.head()


# In[56]:


y=ford.iloc[:,-1]
y.shape
y.head()


# In[58]:


import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=56)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# Linear Regression model::

# In[98]:


from sklearn import linear_model
linear=linear_model.LinearRegression()


# In[100]:


linear.fit(x_train,y_train)


# In[104]:


pred=linear.predict(x_test)
pred
pred.shape


# In[ ]:


linear.coef_


# In[ ]:


linear.intercept_


# In[120]:


R2=linear.score(x_train,y_train)
R2


# In[122]:


Adj_R2=1-(((1-R2)*(14249-1))/(14249-8-1))#[no. of independent train sample =N=14249,no. of independent col=p=8]
Adj_R2


# In[124]:


pred_train=linear.predict(x_train)
pred_train


# In[130]:


pred_train.shape


# In[ ]:


mean_y=y_train.mean()
mean_y


# In[ ]:


SSE=np.sum(np.square(pred_train-y_train))
SSE


# In[ ]:


SSR=np.sum(np.square(pred_train-mean_y))
SSR


# In[ ]:


Rsq=SSR/(SSR+SSE)
Rsq


# In[ ]:


#MAE-Mean Absolute Error
MAE=metrics.mean_absolute_error(pred,y_test)
MAE


# In[ ]:


MSE=metrics.mean_squared_error(pred,y_test)
MSE


# In[144]:


RMSE=np.sqrt(MSE)
RMSE


# In[ ]:


error=pred-y_test
error
error_abs=np.abs(error)
error_abs


# In[ ]:


MAPE=np.mean(error_abs/y_test)*100
MAPE


# In[172]:


Accuracy=(100-MAPE)
Accuracy


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter(pred,y_test)
plt.show()


# In[ ]:


from scipy import stats
slope,intercepts,r,p,std_err=stats.linregress(pred,y_test)
def myfunc(y_test):
 return slope*y_test+intercepts
mymodel=list(map(myfunc,y_test))
plt.scatter(pred,y_test)
plt.plot(y_test,mymodel)
plt.show()


# Ridge model::

# In[154]:


rd=Ridge()
rd.fit(x_train,y_train)
Ridge()


# In[156]:


rd_pred=rd.predict(x_test)
rd_pred


# In[158]:


list(rd.coef_)#rd.coef_=alpha , which is the hyper parmeter, optimum hyper parameter value=1.1166422018169497


# In[160]:


rd_R2=rd.score(x_train,y_train)
rd_R2


# In[162]:


rd_adj_R2=1-(((1-R2)*(14249-1))/(14249-8-1))
rd_adj_R2


# In[ ]:


df_1=pd.DataFrame({"Feature_importances":rd.coef_,"columns":list(x)})
df_1


# In[ ]:


df_2=pd.DataFrame({"Actual":y_test,"Predictions":rd_pred})
df_2


# In[164]:


MSE_rd=metrics.mean_squared_error(rd_pred,y_test)
MSE_rd


# In[166]:


RMSE_rd=np.sqrt(MSE_rd)
RMSE_rd


# In[148]:


MAPE_rd=np.mean(error_abs/y_test)*100
MAPE_rd


# In[150]:


Accuracy_rd=(100-MAPE)
Accuracy_rd


# In[ ]:


sns.lmplot(x="Actual",y="Predictions",data=df_2,fit_reg=False)
d_line=np.arange(df_2.min().min(),df_2.max().max())
plt.plot(d_line,color="red",linestyle="-")
plt.show()


# Polinomial regression model::

# In[60]:


x


# In[62]:


from sklearn.linear_model import LinearRegression


# In[66]:


poly_features=PolynomialFeatures(degree=8)#Create a polynomial feature transformer::degree=no. of columns x
x_poly=poly_features.fit_transform(x)#transform data


# In[68]:


model=LinearRegression()#Create a LinearRegression model
model.fit(x_poly,y)#train the model
y_pred=model.predict(x_poly)#make predictions


# In[86]:


print(y_pred)


# In[132]:


y_pred.shape


# In[180]:


MSE=metrics.mean_squared_error(pred,y_test)
MSE


# In[142]:


RMSE_poly=np.sqrt(MSE)
RMSE_poly


# In[168]:


error_poly=pred-y_test
error_poly
error_abs_poly=np.abs(error)
error_abs_poly


# In[138]:


MAPE_poly=np.mean(error_abs/y_test)*100
MAPE_poly


# In[170]:


Accuracy_poly=(100-MAPE)
Accuracy_poly


# In[ ]:


plt.scatter(x,y)
plt.plot(x,y_pred,color='red')
plt.show()


# GridsearchCrossValidation on RidgeRegression::

# In[ ]:


ridge=Ridge()


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor as ridge
param_grid={'n_estimators':[10,50,100,200],
           'max_depth':[None,5,10,15],
           'min_samples_split':[2,5,10],
           'min_samples_leaf':[1,5,10]
           }
grid_search=GridSearchCV(ridge(),param_grid,cv=5)
grid_search.fit(x_train,y_train)
print("Best Hyper- Parameters:",
grid_search.best_params_)
print("Best Score:",
grid_search.best_score_)
best_model=grid_search.best_estimator_best_model.fit(x_train,y_train)
gd_pred=best_model.predict(x_test)


# 3.#FINAL INSIGHT::with compare to linear , polynomial and ridge regression model the RMSE values are respecteively-
# RMSE=2419.50947,RMSE_poly=2419.50947, RMSE_rd=2419.57228 and accuracy is same for all three above models =83.1222.
# As the accuracy is belonging in the range of generlize model category . Hence I can conclude that linear and polynomial
# both are suitable for Ford car price data modeling .

# 4#.Best Hyper- Parameters: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}
# Best Score (R2): 0.9327563011695075

# In[ ]:




