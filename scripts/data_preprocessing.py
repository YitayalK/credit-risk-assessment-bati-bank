import pandas as pd
import numpy as np
import os

# Load train and test data
train = pd.read_csv('D:/File Pack/Courses/10Acadamey/Week 6/Technical Content/data/train.csv', low_memory=False)
test = pd.read_csv('D:/File Pack/Courses/10Acadamey/Week 6/Technical Content/data/test.csv', low_memory=False)

# Basic overview
print(train.info())
print(train.describe())
print(train.isnull().sum())
