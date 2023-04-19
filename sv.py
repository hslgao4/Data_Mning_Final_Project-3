#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
import pickle #dataframe save and load
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
#%%
df = pd.read_pickle("./df_clean_min.pkl")
#%%
df.info()
#%%
null_counts = df.isnull().sum()
print(null_counts)
#%%
len(df)
#%%
sns.boxplot(data = df, x = 'room_type', y  ='review_scores_rating')
plt.title(" room type Vs review score : World 1")
plt.xlabel("room_type")
plt.ylabel("review Score")

#%%
sns.boxplot(data = df, x = 'beds', y  ='price')
plt.title(" room type Vs review score : World 1")
plt.xlabel("beds")
plt.ylabel("price")
#%%
sns.boxplot(data = df, x = 'accommodates', y  ='price')
plt.title(" room type Vs review score : World 1")
plt.xlabel("accomodates")
plt.ylabel("price")
#%%
sns.scatterplot(data=df,x="review_scores_rating", y="price")
plt.show()
#%%
sns.scatterplot(data=df,x="review_scores_rating", y="price")
plt.show()
#%%
sns.scatterplot(data=df,x="beds", y="price")
plt.show()
#%%
sns.scatterplot(data = df, x = 'review_scores_rating', y  ='avail_365')
#%%
sns.scatterplot(data = df, x = 'price', y  ='avail_365')
#%%
sns.boxenplot(data = df, x = 'instant_bookable', y  ='avail_365')

#%%
sns.boxenplot(data = df, x = 'instant_bookable', y  ='price')

