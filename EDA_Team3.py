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
from scipy.stats import chi2_contingency

#%%[markdown]
### I. Data cleaning
#%%
df = pd.read_csv("listings_220611.csv")
df.id = df.id.astype(int)
df.shape
# %%
df.dtypes
# %%
dt_2206 = pd.read_csv("listings_220611_init.csv")
dt_2206.rename(columns={'id': 'listing_id'},inplace=True)
temp = copy.deepcopy(df)
col = ['listing_id','host_response_time']
temp2 = dt_2206[col]
temp = temp.join(temp2)
df = temp
# %%
# create target variable "review_scores_rating_t2"
dt_2303 = pd.read_csv("listings_230319.csv")
dt_2303.rename(columns={'id': 'listing_id'},inplace=True)
dt_2303['review_scores_rating_t2'] = dt_2303['review_scores_rating'].apply(lambda x: 0 if x >= 4.5 else 1 if x < 4.5 else -9)
dt_2303.review_scores_rating_t2.value_counts(dropna=False,normalize=True).sort_index().mul(100).round(1)
col = ['listing_id','review_scores_rating_t2']
temp2 = dt_2303[col]
df = pd.merge(df, temp2, how='left', on=['listing_id'])

# %%
# create "price_per_room"
df['price_per_room'] = df['price'] / df['bedrooms']
# %%
# transfer data type to bool
def replace_bool(df,var):
    df[var] = df[var].map(lambda x: 1 if x=='t'else 0 if x=='f' else np.nan)

replace_bool(df,'has_availability')
replace_bool(df,'host_has_profile_pic')
replace_bool(df,'host_identity_verified')   
df.shape  
# %%
df = df.rename(columns={'host_acceptance_rate': 'host_accept.R', 'host_listings_count': 'Listings.DC', 'host_total_listings_count': 'Listings.Total', 'host_has_profile_pic': 'profile.pic',
                        'host_identity_verified': 'identity.verify', 'minimum_nights': 'min_nights', 'maximum_nights': 'max_nights', 'has_availability':'availability', 
                         'availability_30': 'avail_30', 'availability_60': 'avail_60', 'availability_90':'avail_90', 'availability_365': 'avail_365'
                             })
# %%
df.columns
# %%
df.drop([
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
       ], axis=1, inplace=True)
df.shape
#%%
df.info()
# %%
df.iloc[:, :16].head()
#%%
df.iloc[:, :32].head()
#%%
print(f'Df Shape: {df.shape}')
print(f'Number of observations: {len(df)}')
# %% [markdown]
##### df is ready fror EDA
#%%[markdown]
### II. EDA
#%%[markdown]
# 1. host_accept.R
# %%
print("Null values:", df['host_accept.R'].isnull().sum())
df_dropna_R = df.dropna(subset=['host_accept.R'])
print(df_dropna_R['host_accept.R'].describe())

#df_dropna_R.to_csv('/Users/lianggao/Desktop/new_dataset.csv', index=False)
plt.hist(df_dropna_R["host_accept.R"], bins=10, alpha=0.5)
plt.title('Histogram plot for Host Accept Rate')
plt.xlabel('Accept Rate')
plt.ylabel('Frequency')
plt.show()

#%%[markdown]
# 2. profile.pic, identity.verify, availability
# %%
def bool_var(df, var):
    print("Information for '", var, "'")
    print("The number of NA: ", df[var].isnull().sum()) 
    df_dropna = df.dropna(subset=[var])
    print(df_dropna[var].describe())

    df_dropna[var].value_counts().plot(kind = 'bar')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.show()

bool_var(df,'profile.pic')
bool_var(df, 'identity.verify')
bool_var(df, 'availability')
bool_var(df, 'instant_bookable')


#%%[markdown]
# 3. accommodates, bedrooms, beds
# %%
def bed_var (df, var):
    print("Information for '", var, "'")
    print("Statistical information: \n", df[var].describe())

    value_counts = df['accommodates'].value_counts()
    sorted_counts = value_counts.sort_index(ascending=True)
    sorted_counts.plot(kind='bar')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.show()

bed_var(df, 'accommodates')
bed_var(df, 'bedrooms')
bed_var(df, 'beds')

#%%[markdown]
# 4. avail_30, avail_60, avail_90, avail_365
# %%
print("Information for 'avail_30'")
print("Statistical information: \n", df['avail_30'].describe())

plt.hist(df["avail_30"], bins = 30, alpha=0.5)
plt.title('Histogram plot for avail_30')
plt.xlabel('avail_30')
plt.ylabel('Frequency')
plt.show()

#%%
print("Information for 'avail_60'")
print("Statistical information: \n", df['avail_60'].describe())

plt.hist(df["avail_60"], bins = 60, alpha=0.5)
plt.title('Histogram plot for avail_60')
plt.xlabel('avail_60')
plt.ylabel('Frequency')
plt.show()
#%%
print("Information for 'avail_90'")
print("Statistical information: \n", df['avail_90'].describe())

plt.hist(df["avail_90"], bins = 60, alpha=0.5)
plt.title('Histogram plot for avail_90')
plt.xlabel('avail_90')
plt.ylabel('Frequency')
plt.show()
# %%
print("Information for 'avail_365'")
print("Statistical information: \n", df['avail_365'].describe())

plt.hist(df["avail_365"], bins = 365, alpha=0.5)
plt.title('Histogram plot for avail_365')
plt.xlabel('avail_365')
plt.ylabel('Frequency')
plt.show()
#%%[markdown]
# 5. price, price_per_room
#%%
print("Information for 'price'")
print(df["price"].describe())
print("\nNull value:", df["price"].isnull().sum())
price = df['price']
plt.hist(price, bins=12)  # Adjust number of bins as needed
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Prices')
plt.show()

#%%
plt.boxplot(df['price'])
plt.xlabel('Price')
plt.title('Box Plot of Prices')
plt.show()
# %%
print("\nInformation for 'price_per_room")
print(df["price_per_room"].describe())
print("\n", df["price_per_room"].isnull().sum())

count = df[df['price_per_room'] > 1000]['price_per_room'].count()
print("\nCount for price_per_room over 1000", count)
df1 = df.drop(df[df['price_per_room'] >= 1000].index)
print(df1["price_per_room"].describe())

#plot before dropping values over 1000
price_per = df['price_per_room']
plt.hist(price_per, bins=12)  # Adjust number of bins as needed
plt.xlabel('Price_per_room before dropping value over 1000')
plt.ylabel('Frequency')
plt.title('Histogram of Price_per_room')
plt.show()

#plot after dropping values over 1000
price_per1 = df1['price_per_room']
plt.hist(price_per1, bins=12)  # Adjust number of bins as needed
plt.xlabel('Price_per_room after drooping')
plt.ylabel('Frequency')
plt.title('Histogram of Price_per_room')
plt.show()
# %% [markdown]
# 6. review_rating_score
# %%
print(df['review_scores_rating'].isnull().sum())
df_dropna = df.dropna(subset=['review_scores_rating'])
print(df_dropna['review_scores_rating'].describe())

# %%
plt.hist(df['review_scores_rating'], density=True, alpha=0.6)
df['review_scores_rating'].plot(kind='kde', linewidth=2)
plt.xlabel('Review Scores Rating')
plt.ylabel('Density')
plt.title('Kernel Density Plot of Review Scores Rating')
plt.show()
# %%
df.boxplot(column='review_scores_rating', by='room_type')
plt.xlabel('Room Type')
plt.ylabel('Review Scores Rating')
plt.title('Box Plot of Review Scores Rating by Room Type')
plt.suptitle('')  # Remove default title
plt.show()
# %% [markdown]
# 7. Try to check the relationship between some variables before building models
# %%
sns.boxplot(data = df, x = 'beds', y  ='price_per_room')
plt.title(" beds Vs price_per_room")
plt.xlabel("beds")
plt.ylabel("price_per_room")
# %%
sns.boxplot(data = df, x = 'accommodates', y  ='price_per_room')
plt.title(" accommodates Vs price_per_room")
plt.xlabel("accomodates")
plt.ylabel("price_per_room")
# %%
sns.scatterplot(data=df,x="beds", y="price_per_room")
plt.show()
# %%
sns.scatterplot(data=df,x="review_scores_rating", y="price_per_room")
plt.show()

#%%
sns.boxenplot(data = df, x = 'instant_bookable', y  ='price_per_room')
#%%
sns.scatterplot(data = df, x = 'price_per_room', y  ='avail_30')
# %%
sns.boxplot(data = df, x = 'room_type', y  ='review_scores_rating')
plt.title(" room type Vs review_scores_rating")
plt.xlabel("room_type")
plt.ylabel("review score")
#%%
sns.scatterplot(data=df,x="review_scores_rating", y="price_per_room")
plt.show()
# %%
sns.scatterplot(data = df, x = 'review_scores_rating', y  ='avail_30')
# %% [markdown]
# 8. Correlation check
# %%
# Selecting relevant columns for correlation analysis
cols_for_correlation = ['review_scores_rating', 'accommodates', 'bedrooms', 'beds', 'price', 'min_nights', 'max_nights', 'availability', 'number_of_reviews']

# Calculating Pearson correlation coefficients
correlation_df = df_dropna[cols_for_correlation].corr(method='pearson')

# Plotting correlation heatmap using seaborn
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0)
plt.title('Pearson Correlation Heatmap')
plt.show()

# %% [markdown]
# 9. Chi-square test
def test_inde (df, x, y):

    contingency_table = pd.crosstab(df[x], df[y])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("\nChi-square test for", x, "and", y)
    print('Chi-square statistic:', chi2)
    print('p-value:', p)
    print('Degrees of freedom:', dof)


test_inde (df_dropna, "review_scores_rating_t2", "profile.pic")
test_inde (df_dropna, "review_scores_rating_t2", "identity.verify")
test_inde (df_dropna, "review_scores_rating_t2", "availability")
test_inde (df_dropna, "review_scores_rating_t2", "instant_bookable")
# %%
