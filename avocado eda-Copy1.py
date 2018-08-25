
# coding: utf-8

# In[2]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

import sklearn
import warnings
warnings.filterwarnings("ignore")


# In[9]:


df = pd.read_csv(r"C:\Users\User\Desktop\avocado.csv\avocado.csv")


# In[23]:


df.head()


# In[20]:


df.info()


# In[97]:


df['AveragePrice'].mean()


# In[98]:


df['Total Volume'].mean()


# In[5]:


df.describe()


# In[7]:


#counting region columns and number of entries per region 

num_list_items = 10
regions = df.groupby(df.region)
print("Total regions : ", len(regions))
print("Printing first ", num_list_items, "Region ")
print("-------------")
for name, group in regions:
    print(name, " : ", len(group))
    num_list_items -= 1
    if num_list_items == 0: break


# In[101]:


#Field partners distribute loans for kiva, this shows the top ones and funding counts

print("Top Avocado buying Regions : ", len(df["region"].unique()))
print(df["region"].value_counts().head(10))
Region = df['region'].value_counts().head(40)
plt.figure(figsize=(15,8))
sns.barplot(region.index, region.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical', fontsize=14)
plt.xlabel('Region', fontsize=18)
plt.ylabel('Total Sales Volime', fontsize=18)
plt.title("Top Avocado buying Regions", fontsize=25)
plt.show()


# In[8]:


#requires calculating total sales in usd

df['Total Sales'].plot(kind = 'hist', rot=70, logx=False, logy=True)
plt.xlabel('total sales volume')
plt.show()


# In[104]:


#Year and Price Boxplot for Regions

mask = df['type']=='conventional'
g = sns.factorplot('AveragePrice','region',data=df[mask],
                   hue='year',
                   size=8,
                   aspect=0.6,
                   palette='Blues',
                   join=False,
              )
plt.show()


# In[105]:


#average price in 2018 calculation
#groupby region and sort by price then grab the indices

order = (
   df[mask & (df['year']==2018)]
    .groupby('region')['AveragePrice']
    .mean()
    .sort_values()
    .index
)


# In[106]:



g = sns.factorplot('AveragePrice','region',data=df[mask],
                   hue='year',
                   size=8,
                   aspect=0.6,
                   palette='Blues',
                   order=order,
                   join=False,
              )


# In[107]:


df['AveragePrice'].plot(kind = 'hist', rot=70, logx=False, logy=True)
plt.xlabel('Average Price')
plt.show()


# In[108]:


df.groupby(['AveragePrice', 'region']).mean()


# In[109]:


df.groupby(['Total Sales', 'region']).size()


# In[110]:


df_index = df.groupby(['Total Sales', 'region']).sizerank().reset_index()
df_index[:25]


# In[10]:


#transform dates into columns for year month day
# feature engineer date so we can use the specifics in our ML model

df['Date']=pd.to_datetime(df['Date'])
df['Month']=df['Date'].apply(lambda x:x.month)
df['Day']=df['Date'].apply(lambda x:x.day)


# In[22]:


#Investigating region by highest price, lowest price and megacity price

regions = ['SanFrancisco', 'Chicago', 'PhoenixTucson']


# In[23]:


mask = (
    df['region'].isin(regions)
    & (df['type']=='conventional')
)


# In[24]:


g = sns.factorplot('Month','AveragePrice', data=df[mask],
               hue='year',
               row='region',
               aspect=2,
               palette='Blues',
              )


# In[30]:


#transform dates into columns for year month day
# feature engineer date so we can use the specifics in our ML model

df['Date']=pd.to_datetime(df['Date'])
df['Month']=df['Date'].apply(lambda x:x.month)
df['Day']=df['Date'].apply(lambda x:x.day)


# In[31]:


#price of avocados over time
#how to add axes labels?
#would love to do this by region...

byDate=df.groupby('Date').mean()
plt.figure(figsize=(12,8))
byDate['AveragePrice'].plot()
plt.title('Average Price')


# In[32]:


df['region'].nunique()


# In[29]:


df['type'].nunique()


# In[33]:


df.head()


# In[27]:


#Correlation Matrix - values more then 0.05 should be investigated
corr = df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation between features')
corr


# In[ ]:


#Correlation matrix shows that average price is not strongly correlated with any of the features but they appear correlated with each other, 

#Indicates further engineering on the categorical variable - region


# In[11]:


#total volume by average price
sns.jointplot(x='Total Volume', y='AveragePrice', data=df, color='blue')


# In[12]:


#total volume by large bags
sns.jointplot(x='Total Volume', y='Large Bags', data=df, color='blue')


# In[127]:


#small bags by volume
sns.jointplot(x='Total Volume', y='Small Bags', data=df, color='blue')


# In[13]:


sns.jointplot(x='Total Volume', y='Total Bags', data=df, color='blue')


# In[48]:


#distribution of average prices 

fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.boxplot(x='year',y='AveragePrice',data=df,color='blue')


# In[47]:


fig, ax = plt.subplots(1, 1, figsize=(10,6))
price_val = df['AveragePrice'].values
sns.distplot(price_val, color='b')
ax.set_title('Distribution of Average Price', fontsize=14)
ax.set_xlim([min(price_val), max(price_val)])


# In[34]:


#date is being dropped as its been transformed 
#region is being dropped for this model - how do we manage this type of feature?

df_final=pd.get_dummies(df.drop(['region','Date'],axis=1),drop_first=True)


# In[35]:


df_final.head()


# In[36]:


#Average price is a continous variable, linear regression
#X array is the training feature
#Y array is the target variable - Average price

X=df_final.iloc[:,1:14]
y=df_final['AveragePrice']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[37]:


#create and train the model 

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)


# In[38]:


#scatter of trained regression model, not the best model

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[39]:


plt.scatter(x=y_test, y=pred)


# In[40]:


#decision tree regressor model 

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
pred=dtr.predict(X_test)


# In[41]:


plt.scatter(x=y_test,y=pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[42]:


print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[43]:


from sklearn.ensemble import RandomForestRegressor
rdr = RandomForestRegressor()
rdr.fit(X_train,y_train)
pred=rdr.predict(X_test)


# In[44]:


print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[45]:


sns.distplot((y_test-pred),bins=50)


# In[46]:


data = pd.DataFrame({'Y Test':y_test , 'Pred':pred},columns=['Y Test','Pred'])
sns.lmplot(x='Y Test',y='Pred',data=data,palette='rainbow')
data.head()

