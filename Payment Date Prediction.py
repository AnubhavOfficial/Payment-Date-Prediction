#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING LIBRARIES 

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")
pd. set_option('display.max_columns', 25) # or 1000.


# ### DATA PREPROCESSING

# In[78]:


#Importing dataset
data = pd.read_csv('H2HBABBA2262.csv')


# In[79]:


#Filtering columns on the basis of null values or completely unique values
data.nunique()


# #### Observations

# 
#      1) posting_id has only 1 value  i.e Constant column
# 
#      2) area business has all 0/null values  i.e Null column                 
#      
#      3) invoice_id has all unique values except 7  i.e Unique column
#      
#      4) doc_id has all unique values  i.e unique column
#      
#      5)isOpen has 2 uniques values i.e it only tells about clear date. whether clear date is there or not
#      
#      6)Document_type has 2 unique values 

# In[80]:


#checking count of all the rows with null values 
data.isna().sum()


# In[81]:


#dropping all the duplicate rows as they are of no use
data = data.drop_duplicates(keep='first')


# ### Null Imputation

# In[82]:


#checking all the null values of Invoice_id
data[data.invoice_id.isnull()]


# In[83]:


#checking all rows with document_type as X2 for futher observation
data[data['document type'].isin(['X2'])]


# #### Observation

# 1) Document type is X2 only for those rows which has null invoice ID
# 
# 2) Invoice Id is missing which may be a human error, as the payment has been done by the customer. So we will keep these rows as    important information is not missing

# In[84]:


#dropping all the columns that don't contribute much in prediction and dont give information and no pattern can be seen in them
data = data.drop(columns=['posting_id','area_business','doc_id','isOpen','document type'])


# In[85]:


data


# 
# 
# 
#     We will drop document_create_date as we will consider document_create_date.1 because it is normalized.                        { Given in the Data Dictionary Tech Track }
#     
#     Also removing baseline_create_date as it is highly correlated with document_create_date.1 and almost all the values are same

# In[86]:


#dropped document create date and baseline create date
data.drop(columns =['document_create_date','baseline_create_date'],inplace=True)


# In[87]:


data.shape


# In[88]:


#removed 1204 duplicated rows 
data.shape


# In[89]:


#again checking the null values
data.isna().sum()


# In[90]:


#checking the data types of all the columns
data.info()


# In[91]:


#converting all the dates column in date time format
data['clear_date'] = pd.to_datetime(data['clear_date'], format = '%Y-%m-%d')
data['posting_date'] = pd.to_datetime(data['posting_date'], format = '%Y-%m-%d')
data['document_create_date.1'] = pd.to_datetime(data['document_create_date.1'], format = '%Y%m%d')
data['due_in_date'] = pd.to_datetime(data['due_in_date'], format = '%Y%m%d')


# In[92]:


#checking all the updates of the date format
data


# In[93]:


#counting the null value count in
data['clear_date'].isnull().value_counts()


# In[94]:


#
data['business_code'] = np.where(data['business_code'].isin([ "U002",
                                                        "U005",
                                                        "U007"]),"Other codes",data['business_code'])


# In[95]:


#Label encoding on business_code to convert it to int type in all 
#Doing it here as it will save alot of effot and time, otherwise we had to do labelling seperately on main_train and main_test

from sklearn.preprocessing import LabelEncoder

bcode_enc = LabelEncoder()
bcode_enc.fit(data['business_code'])
data['bcode_enc'] = bcode_enc.transform(data['business_code'])


cus_name_enc = LabelEncoder()
cus_name_enc.fit(data['name_customer'])
data['cus_name_enc'] = cus_name_enc.transform(data['name_customer'])


term_enc = LabelEncoder()
term_enc.fit(data['cust_payment_terms'])
data['cus_terms_enc'] = term_enc.transform(data['cust_payment_terms'])


# In[96]:


#segregating the dataset into main_train and main_test based on the null value of clear_date
#main_train has all the rows with given clear_date value
#main_test has all the rows with null clear_date value


main_train = data[data['clear_date'].isnull()==False]
main_test = data[data['clear_date'].isnull()]


# In[97]:


main_train.shape  , main_test.shape


# In[98]:


main_train.info()


# In[99]:


#sorting values by posting_date as the delay is dependent on the posting date
# we only use past data to train the model , hence sorting ensures that while splittig the data only past data is there on the train set

main_train = main_train.sort_values(by='posting_date')


# In[100]:


#Making the target Variable, delay in our case
main_train['Delay'] = (main_train['clear_date'] - main_train['due_in_date']).dt.days


# In[101]:


main_train


# ## DATA SPLITTING

# In[102]:


# Converting the main_train into X and y so that we can pass it onto train_test_split function

# X --> contains the dataframe without the target i.e delay
X = main_train.drop('Delay',axis=1)

# y --> contains only the target value 
y = main_train['Delay']


# In[103]:


#splitting the data first into two part -- doing a 70:30 split i.e 30% data fed to intermediate test data set
from sklearn.model_selection import train_test_split
X_train,X_inter_test,y_train,y_inter_test = train_test_split(X,y,test_size=0.3,random_state=0 , shuffle = False)


# In[104]:


#Further splitting the X_inter_test,y_inter_test into X_val,X_test,y_val,y_test 
#doing a 50:50 split i.e 50% data fed to test data set and rest 50% to validation set
X_val,X_test,y_val,y_test = train_test_split(X_inter_test,y_inter_test,test_size=0.5,random_state=0 , shuffle = False)


# In[105]:


X_train.shape , X_val.shape , X_test.shape


# # EDA

# In[106]:


# distribution of the target column i.e. Delay
# right skewed distribution
sns.distplot(y_train)


# In[107]:


#No direct trend
sns.scatterplot(data=X_train.merge(y_train,on = X_train.index), x="Delay", y="due_in_date")


# In[108]:


#plotting scatter plot 
sns.scatterplot(data=X_train.merge(y_train,on = X_train.index), x="Delay", y="total_open_amount")


# In[109]:


#checking the different parameters of the X_train
X_train.describe()


# In[110]:


#visual representation of the relation between different columns of our data pairwise
sns.pairplot(X_train)


# # Feature Engg

# In[111]:


#Checking the data types
X_train.info()


# In[112]:


# For catagorical columns with relatively low unique value (<= 15) -- looking for value_counts
# if unique count ==1 , constant column 

for col in X_train.columns:
    if X_train[col].nunique()<= 15:
        print(X_train[col].value_counts())
        print('#########################')


# In[113]:


#removing business year as it is a constant column
X_train.drop('buisness_year',inplace=True,axis=1)
X_test.drop('buisness_year',inplace=True,axis=1)
X_val.drop('buisness_year',inplace=True,axis=1)


# In[114]:


#converting the total amount in one currency in all test,val and train
X_train['amt_same_curr'] = np.where(X_train.invoice_currency=="CAD",X_train.total_open_amount*0.81337,X_train.total_open_amount)
X_test['amt_same_curr'] = np.where(X_test.invoice_currency=="CAD",X_test.total_open_amount*0.81337,X_test.total_open_amount)
X_val['amt_same_curr'] = np.where(X_val.invoice_currency=="CAD",X_val.total_open_amount*0.81337,X_val.total_open_amount)


# In[115]:


#we dont need invoice currency and total open amount as we have created a seperate column amt_same_curr using these two.
#removing from test, val and train
X_train.drop(['invoice_currency','total_open_amount'],inplace=True,axis=1)
X_test.drop(['invoice_currency','total_open_amount'],inplace=True,axis=1)
X_val.drop(['invoice_currency','total_open_amount'],inplace=True,axis=1)


# In[116]:


X_train.info()


# In[117]:


#Extracting the day, month and weekday from the due_in_date to convert it into int
X_train['due_month'] = X_train['due_in_date'].dt.month
X_train['due_day'] = X_train['due_in_date'].dt.day
X_train['due_weekday'] =X_train['due_in_date'].dt.weekday

X_test['due_month'] = X_test['due_in_date'].dt.month
X_test['due_day'] = X_test['due_in_date'].dt.day
X_test['due_weekday'] =X_test['due_in_date'].dt.weekday

X_val['due_month'] = X_val['due_in_date'].dt.month
X_val['due_day'] = X_val['due_in_date'].dt.day
X_val['due_weekday'] =X_val['due_in_date'].dt.weekday


# In[118]:


#Calculating the time given to the customer for making the payment by subtracting posting_date from due_in_date
X_train['time_given'] = (X_train['due_in_date'] - X_train['posting_date']).dt.days
X_test['time_given'] = (X_test['due_in_date'] - X_test['posting_date']).dt.days
X_val['time_given'] = (X_val['due_in_date'] - X_val['posting_date']).dt.days


# In[119]:


#Looking for other data type conversion
X_train.info()


# # Feature Selection

# In[120]:


#Selecting these particular features and making a list of the same
features = ['bcode_enc','cus_name_enc','cus_terms_enc','amt_same_curr','due_month','due_day','time_given','due_weekday']


# In[121]:


#verifying the features
X_train[features]


# In[122]:


#Plotting a heatmap to check further correlation between target column and other olumns
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(X_train.merge(y_train , on = X_train.index ).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# # Modelling

# In[123]:


#Creating a DecisionTree model for this dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0 , max_depth=10)


# In[124]:


#regressor is the name of the model
#Training the model
regressor.fit(X_train[features], y_train)


# In[125]:


#predicting the delay on test dataset
y_predict2 = regressor.predict(X_test[features])


# In[126]:


#Calculating the mean squared error
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_predict2, squared=False)


# In[127]:


#Finding the accuracy of the model
random_model_accuracy = round(regressor.score(X_test[features], y_test)*100,2)
print(round(random_model_accuracy,2),'%')


# In[128]:


#Calculating the MSE for train and validation dataset
y_pred_train = regressor.predict(X_train[features])
y_pred_validation = regressor.predict(X_val[features])


# In[129]:


from sklearn.metrics import mean_squared_error
train_rmse = mean_squared_error(y_train, y_pred_train)**0.5
validation_rmse = mean_squared_error(y_val, y_pred_validation)**0.5
print(f'Train {train_rmse}')
print(f'Validation {validation_rmse}')


# In[130]:


#Observing the values of the predicted delay corresponding the the actual delay
met = pd.DataFrame(zip(y_predict2 , y_test),columns=['Predicted','Actuals'])


# In[131]:


met.head(10)


# In[132]:


X_train


# In[133]:



#bucketisation function

def bucketisation(value):
    if value > 60:
        return ">60 days"
    elif value > 45:
        return "46-60 days"
    elif value > 30:
        return "31-45 days"
    elif value > 15:
        return "16-30 days"
    elif value >=0:
        return "0-15 days"
    elif value <0:
        return "Advance Payment"


# In[134]:


#Cleaning and preprocessing the main_test 
main_test.info()


# In[135]:


features


# In[136]:


#Making all the required columns from main_test
main_test['amt_same_curr'] = np.where(main_test.invoice_currency=="CAD",main_test.total_open_amount*0.81337,main_test.total_open_amount)


# In[137]:


#Making all the required columns from main_test
main_test['due_month'] = main_test['due_in_date'].dt.month
main_test['due_day'] = main_test['due_in_date'].dt.day
main_test['due_weekday'] =main_test['due_in_date'].dt.weekday
main_test['time_given'] = (main_test['due_in_date'] - main_test['posting_date']).dt.days


# In[138]:


#Adding predicted delay column to the main_test
main_test['delay'] = regressor.predict(main_test[features])
main_test['delay'] = main_test['delay'].apply(np.ceil)


# In[139]:


#adding predicted delay with the due_in_date to find predicted payment date
main_test['predicted_payment_date'] = main_test['due_in_date'] + main_test['delay'].apply(lambda x: pd.Timedelta(x,unit='d'))


# In[140]:


main_test


# In[141]:


# bucketisation of data


X_train["Aging_Bucket"] = y_train.apply(bucketisation)
X_val["Aging_Bucket"] = y_val.apply(bucketisation)
X_test["Aging_Bucket"] = y_test.apply(bucketisation)

main_test["Aging_Bucket"] = main_test['delay'].apply(bucketisation)


# In[142]:


#We can now see predicted date, delay and aging bucket is added to main_test dataset
main_test.shape


# In[143]:


main_test.drop(columns = features, inplace=True,axis=1)


# In[144]:


main_test


# # THE END
