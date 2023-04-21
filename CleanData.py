# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
# %%
df = pd.read_csv("listings_220611.csv")
df.shape

# %%
# Creat "price per room and move the position"
df['price_per_room'] = df['price'] / df['bedrooms']
cols = df.columns.tolist()
cols.insert(14, cols.pop(-1))
df = df[cols]

# %%
def replace_bool(df,var):
    df[var] = df[var].map(lambda x: 1 if x=='t'else 0 if x=='f' else np.nan)

replace_bool(df,'has_availability')
replace_bool(df,'host_has_profile_pic')
replace_bool(df,'host_identity_verified')  
replace_bool(df, 'instant_bookable')


# %%
df = df.rename(columns={'host_acceptance_rate': 'host_accept.R', 'host_listings_count': 'Listings.DC', 'host_total_listings_count': 'Listings.Total', 'host_has_profile_pic': 'profile.pic',
                        'host_identity_verified': 'identity.verify', 'minimum_nights': 'min_nights', 'maximum_nights': 'max_nights', 'has_availability':'availability', 
                         'availability_30': 'avail_30', 'availability_60': 'avail_60', 'availability_90':'avail_90', 'availability_365': 'avail_365'
                             })


# %%
df.drop(['calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms'], axis=1, inplace=True)
df.shape

# %%
df.iloc[:, :17].head()
# %%
df.iloc[:, 17:35].head()
# %%
print(f'Df Shape: {df.shape}')
print(f'Number of observations: {len(df)}')

#%%[markdown]
# 1. host_accept.R
# %%
print(df['host_accept.R'].isnull().sum())
df_dropna_R = df.dropna(subset=['host_accept.R'])
print(df_dropna_R['host_accept.R'].describe())

#df_dropna_R.to_csv('/Users/lianggao/Desktop/new_dataset.csv', index=False)
plt.hist(df_dropna_R["host_accept.R"], bins=10, alpha=0.5)
plt.title('Histogram plot for Host Accept Rate')
plt.xlabel('Accept Rate')
plt.ylabel('Frequency')
plt.show()

#%%[markdown]
# For bool variables
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



# %%

print("Information for 'avail_30'")
print("Statistical information: \n", df['avail_30'].describe())

plt.hist(df["avail_30"], bins = 30, alpha=0.5)
plt.title('Histogram plot for avail_30')
plt.xlabel('avail_30')
plt.ylabel('Frequency')
plt.show()

# %%
print("Information for 'avail_60'")
print("Statistical information: \n", df['avail_60'].describe())

plt.hist(df["avail_60"], bins = 60, alpha=0.5)
plt.title('Histogram plot for avail_60')
plt.xlabel('avail_60')
plt.ylabel('Frequency')
plt.show()
# %%
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

# %%
print(df["price_per_room"].describe())
print(df["price_per_room"].isnull().sum())
df = df.drop(df[df['price_per_room'] >= 1000].index)
#count = df[df['price_per_room'] > 1000]['price_per_room'].count()

print(df["price_per_room"].describe())
#sns.boxplot(x='price_per_room', data=df)
sns.violinplot(x='price_per_room', data=df)
# %%
print(df['review_scores_rating'].isnull().sum())

df_dropna = df.dropna(subset=['review_scores_rating'])
print(df_dropna['review_scores_rating'].describe())

# %%
df_dropna.loc[df_dropna['review_scores_rating'] <= 3, 'review_scores_rating'] = 3

mask = df_dropna['review_scores_rating'] <= 4.5
df_dropna.loc[mask, 'review_rating'] = 0
df_dropna.loc[~mask, 'review_rating'] = 1

df_dropna['review_rating'].head()
# %%
def test_inde (df, x, y):

    contingency_table = pd.crosstab(df[x], df[y])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("\nChi-square test for", x, "and", y)
    print('Chi-square statistic:', chi2)
    print('p-value:', p)
    print('Degrees of freedom:', dof)


# %%

# %%
