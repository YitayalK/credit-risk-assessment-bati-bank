import pandas as pd
import numpy as np


# Load train and test data
train = pd.read_csv('D:/File Pack/Courses/10Acadamey/Week 6/Technical Content/data/train.csv', low_memory=False)
test = pd.read_csv('D:/File Pack/Courses/10Acadamey/Week 6/Technical Content/data/test.csv', low_memory=False)

# Ensure TransactionStartTime is in datetime format
train['TransactionStartTime'] = pd.to_datetime(train['TransactionStartTime'])
test['TransactionStartTime'] = pd.to_datetime(test['TransactionStartTime'])

# Create aggregate features per customer
agg_funcs = {
    'Amount': ['sum', 'mean', 'std'],
    'TransactionId': 'count'
}
customer_agg = train.groupby('CustomerId').agg(agg_funcs)
customer_agg.columns = ['Total_Amount', 'Avg_Amount', 'Std_Amount', 'Transaction_Count']
customer_agg = customer_agg.reset_index()

print(customer_agg.head())

#Extract Date/Time Features
def extract_time_features(df):
    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year
    return df

train = extract_time_features(train)
test = extract_time_features(test)

#Encode Categorical Variables
from sklearn.preprocessing import LabelEncoder

# Example for Label Encoding for ProviderId and ChannelId
le_provider = LabelEncoder()
train['ProviderId_encoded'] = le_provider.fit_transform(train['ProviderId'])
test['ProviderId_encoded'] = le_provider.transform(test['ProviderId'])

le_channel = LabelEncoder()
train['ChannelId_encoded'] = le_channel.fit_transform(train['ChannelId'])
test['ChannelId_encoded'] = le_channel.transform(test['ChannelId'])

#Normalization/Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train[['Total_Amount', 'Avg_Amount', 'Std_Amount']] = scaler.fit_transform(
    train[['Amount', 'Amount', 'Amount']]  # adjust according to the features used
)
# Apply similar transformation to test set if needed.

#Handling Missing Values
train.fillna(train.mean(), inplace=True)