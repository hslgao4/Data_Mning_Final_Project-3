# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
df = pd.read_csv("listings_220611.csv")
df.shape

# %%
# Creat "price per room and move the position"
df['price_per_room'] = df['price'] / df['bedrooms']
cols = df.columns.tolist()
cols.insert(14, cols.pop(-1))
df = df[cols]
df.shape
# %%
def replace_bool(df,var):
    df[var] = df[var].map(lambda x: 1 if x=='t'else 0 if x=='f' else np.nan)

replace_bool(df,'has_availability')
replace_bool(df,'host_has_profile_pic')
replace_bool(df,'host_identity_verified')  
replace_bool(df, 'instant_bookable')
df.shape  

# %%
df = df.rename(columns={'host_acceptance_rate': 'host_accept.R', 'host_listings_count': 'Listings.DC', 'host_total_listings_count': 'Listings.Total', 'host_has_profile_pic': 'profile.pic',
                        'host_identity_verified': 'identity.verify', 'minimum_nights': 'min_nights', 'maximum_nights': 'max_nights', 'has_availability':'availability', 
                         'availability_30': 'avail_30', 'availability_60': 'avail_60', 'availability_90':'avail_90', 'availability_365': 'avail_365'
                             })
df.columns

# %%
df.drop(['calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms'], axis=1, inplace=True)
df.columns

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

# %%
