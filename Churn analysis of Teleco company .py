#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np 
import pandas as pd 
import seaborn as sns  
import matplotlib.pyplot as plt


# In[10]:


df = pd.read_csv('Telco-Customer-Churn.csv')

# Display the first few rows of the dataframe
print(df.head())

# Get a summary of the dataset
print(df.info())

df


# There are no missing values

# In[39]:


df.shape


# In[40]:


# Checking the data types of all the columns
df.dtypes


# In[41]:


# Check the descriptive statistics of numeric variables
df.describe()


# # Data Cleaning 

# Create a copy of base data for manupulation & processing

# In[42]:


df_copy=df.copy()


# In[44]:


#totalcharges column has object data type that i change to numeric 
df.TotalCharges=pd.to_numeric(df.TotalCharges,errors='coerce')


# In[45]:


df.dtypes


# In[46]:


df.isnull().sum()


# As we can see there are 11 missing values in TotalCharges column after changing dtatatye. Let's check these records

# In[47]:


df[df['TotalCharges'].isnull()]


# In[48]:


#missing percentage of telecom column 
(df['TotalCharges'].isnull().sum()/7043)*100


# In[49]:


numerical_columns=df[['tenure','MonthlyCharges','TotalCharges']]
numerical_columns


# In[50]:


import pandas as pd
import numpy as np


# Step 1: Identify the numerical columns
numerical_columns = df.select_dtypes(include=[np.number])

# Step 2: Handle non-numeric values by converting them to NaN and then to appropriate numeric types
for col in numerical_columns.columns:
    numerical_columns[col] = pd.to_numeric(numerical_columns[col], errors='coerce')

# Step 3: Compute the correlation matrix
correlation_matrix = numerical_columns.corr()

# Display the first column of the correlation matrix
print(correlation_matrix.iloc[:, 0])


# # use kmn imputer to fill the missing values

# In[51]:


from sklearn.impute import KNNImputer


# In[52]:


# Initialize the KNNImputer with the desired number of neighbors
imputer = KNNImputer(n_neighbors=5)


# In[53]:


# Initialize the KNNImputer with the desired number of neighbors
imputer = KNNImputer(n_neighbors=5)

# Select the columns to impute
columns_to_impute = ['TotalCharges']

# Fit and transform the data to impute missing values
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# Verify that there are no more missing values in TotalChrg
print(df['TotalCharges'].isna().sum())


# ### check if correlation is ok after filling missing data

# In[54]:


import pandas as pd
import numpy as np


# Step 1: Identify the numerical columns
numerical_columns = df.select_dtypes(include=[np.number])

# Step 2: Handle non-numeric values by converting them to NaN and then to appropriate numeric types
for col in numerical_columns.columns:
    numerical_columns[col] = pd.to_numeric(numerical_columns[col], errors='coerce')

# Step 3: Compute the correlation matrix
correlation_matrix = numerical_columns.corr()

# Display the first column of the correlation matrix
print(correlation_matrix.iloc[:, 0])


# ### Finding resaon behind churn

# In[55]:


import seaborn as sns
import matplotlib.pyplot as plt

# Churn distribution by contract type
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.show()

sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Churn by InternetService type')
plt.show()

sns.countplot(x='OnlineSecurity', hue='Churn', data=df)
plt.title('Churn by OnlineSecurity ')
plt.show()

df['MonthlyCharges_bin'] = pd.cut(df['MonthlyCharges'], bins=10)

# Plot the data
plt.figure(figsize=(14, 8))
sns.countplot(data=df, x='MonthlyCharges_bin', hue='Churn')
plt.title('Distribution of Churn by Monthly Charges')
plt.xlabel('Monthly Charges Bin')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.show()






# # To check whether Monthly price is reason for churn or not

# In[57]:


import pandas as pd

# Create the new column 'PriceCategory'
df['Monthly PriceCategory'] = pd.cut(df['MonthlyCharges'],
                             bins=[0, 60, 120],
                             labels=['Low Priced',  'High Priced'],
                             right=False)

# Display the DataFrame
print(df)


# In[58]:


# If the column name is different, replace 'Churn' with the actual column name.

# Group by 'PriceCategory' and calculate churn percentage
churn_percentage = df.groupby('Monthly PriceCategory')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)

print(churn_percentage)


# In[61]:


import matplotlib.pyplot as plt


# Group by 'PriceCategory' and calculate churn percentage
churn_percentage = df.groupby('Monthly PriceCategory')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)

# Plotting
churn_percentage.plot(kind='bar', color='skyblue')
plt.title('Churn Percentage by Price Category')
plt.xlabel('Price Category')
plt.ylabel('Churn Percentage (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ## It seems that monthly price is the reason for churn

# ## Hypothesis test to see if i am correct

# In[62]:


# Filter the DataFrame to get rows where churn is 'Yes' and extract MonthlyCharges
monthly_charges_churn_yes = df[df['Churn'] == 'Yes']
yes= monthly_charges_churn_yes['MonthlyCharges']

monthly_charges_churn_yes = df[df['Churn'] == 'No']
No= monthly_charges_churn_yes['MonthlyCharges']


# In[65]:


#choosing 30 samples for testing
import numpy as np
import seaborn as sns
samples=[]
for i in range(30):
    samples.append(yes.sample(62).values.tolist())
    
    
samples=np.array(samples)
sampling_mean_yes=samples.mean(axis=1)



sns.histplot(sampling_mean_yes, kde=True)
plt.title('Distribution of sample of churned monthly charges')
plt.show()


# In[66]:


import numpy as np
import seaborn as sns
samples=[]
for i in range(30):
    samples.append(No.sample(62).values.tolist())
    
    
samples=np.array(samples)
sampling_mean_no=samples.mean(axis=1)


sns.histplot(sampling_mean_no, kde=True)
plt.title('Distribution of sample of non_churn Monthly Charges')
plt.show()


# In[67]:


sample_mean_of_churned = sampling_mean_yes.mean()
sample_mean_of_churned


# In[68]:


sample_mean_of_nonchurned = sampling_mean_no.mean()
sample_mean_of_nonchurned


# In[69]:


std_dev_of_churned=np.std(sampling_mean_yes)
std_dev_of_churned


# In[70]:


std_dev_of_nonchurned=np.std(sampling_mean_no)
std_dev_of_nonchurned


# In[72]:


#Null Hypothesis (H0): There is a significant difference in the average monthly charges between customers who have churned and those who have not churned.
#Alternative Hypothesis (H1): There is no significant difference in the average monthly charges between customers who have churned and those who have not churned.
from scipy.stats import t
t_value=13.1
f= 58
cdf_value=t.cdf(t_value,f)
print(cdf_value*2)


# In[ ]:


#p value is 2.0  which is greater than 0.05 so we cannot reject the null hypothesis that There is a significant difference in the average monthly charges between customers who have churned and those who have not churned.


# ## Finding Risky customer

# In[73]:


import pandas as pd

# First, create the 'risk level' column based on the 'Churn' column
df['risk level'] = df['Churn'].apply(lambda x: 'risky' if x == 'Yes' else 'not risky')

# Define the thresholds for high, medium, and low risk levels for risky customers
# You may need to adjust these thresholds based on your specific needs
high_risk_threshold = 80
low_risk_threshold = 40

# Apply the risk levels for 'risky' customers based on MonthlyCharges
def determine_risk_level(row):
    if row['risk level'] == 'risky':
        if row['MonthlyCharges'] >= high_risk_threshold:
            return 'high risky'
        elif row['MonthlyCharges'] <= low_risk_threshold:
            return 'low risky'
        else:
            return 'risky'
    else:
        return row['risk level']

df['risk level'] = df.apply(determine_risk_level, axis=1)

df.head(40)


# In[74]:


df.to_csv('telecom churn analysis data',index= False)

