#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries and packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Reading the dataset
dataset=pd.read_excel("C:/Users/Pavana Sree/Downloads/HousePricePrediction.xlsx")


# In[3]:


print(dataset.head(5))


# In[4]:


dataset.shape


# In[5]:


dataset.dtypes


# In[6]:


obj=(dataset.dtypes=='object')
object_cols=list(obj[obj].index)
print("Categorical variables:",len(object_cols))


# In[7]:


int_=(dataset.dtypes=='int64')
num_cols=list(int_[int_].index)
print("Integer variables:",len(num_cols))


# In[8]:


f1=(dataset.dtypes=='float')
f1_cols=list(f1[f1].index)
print("Float variables:",len(f1_cols))


# In[9]:


numerical_dataset=dataset.select_dtypes(include=['number'])


# In[10]:


#heatmap to identify the relation between the numeric data
plt.figure(figsize=(12,6))
sns.heatmap(numerical_dataset.corr(),
            cmap='viridis',fmt='.2f',
            linewidths=2,
            annot=True)


# In[11]:


unique_values=[]
for col in object_cols:
    unique_values.append(dataset[col].unique().size)


# In[12]:


#Unique values of each object column
plt.figure(figsize=(10,6))
plt.title('No. Unique values of categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)


# In[13]:


#distribution of each unique value 
plt.figure(figsize=(18,36))
plt.title("Categorical Feature distribution: Distribution")
plt.xticks(rotation=90)
index=1

for col in object_cols:
    y=dataset[col].value_counts()
    plt.subplot(11,4,index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index),y=y)
    index+=1


# In[14]:


#dealing with null and missing values
dataset.drop(['Id'],axis=1,inplace=True)


# In[15]:


dataset['SalePrice']=dataset['SalePrice'].fillna(dataset['SalePrice'].mean())


# In[16]:


new_dataset=dataset.dropna()


# In[17]:


new_dataset.isnull().sum()


# In[18]:


from sklearn.preprocessing import OneHotEncoder

s=(new_dataset.dtypes=='object')
object_cols=list(s[s].index)
print("Categorical Variables:")
print(object_cols)
print("No. of Categorical features:",
     len(object_cols))


# In[19]:


#converting the object data into numerical data
OH_encoder=OneHotEncoder(sparse=False,handle_unknown='ignore')
OH_cols=pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index=new_dataset.index
OH_cols.columns=OH_encoder.get_feature_names_out()
df_final=new_dataset.drop(object_cols,axis=1)
df_final=pd.concat([df_final,OH_cols],axis=1)


# In[20]:


df_final


# In[21]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# In[22]:


#training and testing dataset
x=df_final.drop(['SalePrice'],axis=1)
y=df_final['SalePrice']


# In[23]:


x_train,x_valid,y_train,y_valid=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)


# In[27]:


#Trying to choose best algorithm
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

model_SVR=svm.SVR()
model_SVR.fit(x_train,y_train)
y_pred=model_SVR.predict(x_valid)

mean_absolute_percentage_error(y_valid,y_pred)


# In[25]:


from sklearn.ensemble import RandomForestRegressor

model_RFR=RandomForestRegressor(n_estimators=10)
model_RFR.fit(x_train,y_train)
y_pred=model_RFR.predict(x_valid)

mean_absolute_percentage_error(y_valid,y_pred)


# In[26]:


from sklearn.linear_model import LinearRegression

model_LR=LinearRegression()
model_LR.fit(x_train,y_train)
y_pred=model_LR.predict(x_valid)

mean_absolute_percentage_error(y_valid,y_pred)


# In[28]:


from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

cb_model=CatBoostRegressor()
cb_model.fit(x_train,y_train)
preds=cb_model.predict(x_valid) 

cb_r2_score=r2_score(y_valid, preds)
cb_r2_score


# In[ ]:




