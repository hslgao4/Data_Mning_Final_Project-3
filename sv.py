#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
import pickle 
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
#%%
df = pd.read_pickle("D:/Data_Mning_Final_Project-3/df_clean_min.pkl")
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
#%%
cols_for_correlation = ['review_scores_rating','avail_365', 'accommodates', 'bedrooms', 'beds', 'price','availability', 'number_of_reviews', 'reviews_per_month']

# Calculating Pearson correlation coefficients
correlation_df = df[cols_for_correlation].corr(method='pearson')

# Plotting correlation heatmap using seaborn
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0)
plt.title('Pearson Correlation Heatmap')
plt.show()

#%%
df.info()
#%%
df['room_type'].unique()
#%%
from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()
#%%
# Apply label encoding to 'room_type' column
df['room_type_encoded'] = label_encoder.fit_transform(df['room_type'])
#%%
#ONE HOT ENCODING
df = pd.get_dummies(df, columns=['room_type'], prefix='dm')
#%%
df['dm_Entire home/apt']
#%%
df.shape
#%%
df.isna().sum()
#%%
df = df.dropna()
#%%
df.isnull().sum()
#%%
sns.displot(data = df, x ='avail_365')
plt.show()
#%%
sns.displot(data = df, x ='price')
plt.show()
#%%
X = df.drop(['review_scores_rating_t2','listing_id'], axis=1) #All input

ip=['bedrooms','beds','price','min_nights','max_nights','availability','avail_365','number_of_reviews','Avg_neg_review_comment_score','Avg_pos_review_comment_score','Avg_neu_review_comment_score']
#%%
X = X[ip]
y = df['review_scores_rating_t2']
#%%
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=300)
model_fit = model.fit(X_train, y_train)
#%%

print('model accuracy for test', model_fit.score(X_test, y_test))
print('model accuracy for train', model_fit.score(X_train, y_train))
#%%
