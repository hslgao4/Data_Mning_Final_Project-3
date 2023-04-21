# %%
import numpy as np
import pandas as pd
df = pd.read_csv("listings_220611.csv")
df.shape

# %%
df['price_per_room'] = df['price'] / df['bedrooms']
df.shape
# %%
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
df.columns

#%%[markdown]
### Here, look here! ! !
# I am not sure if we need them (the next dropped columns) or not, if we need, just delete that drop
#
# And if you need to add or delete anything, just change the code
# %%
df.drop(['calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms'], axis=1, inplace=True)
df.columns
# %%
df.shape
# %%
df.iloc[:, :17].head()
# %%
df.iloc[:, 17:35].head()
# %%
