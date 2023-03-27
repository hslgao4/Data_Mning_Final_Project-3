#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import copy
import os

#%%[markdown]
'''
## Analysis for Air bnb data <br>
### - Team3
 
#### `1. Raw data sourcing`
# Air bnb Host data in washington dc
listings_220319
listings_2206112
listings_220914
listings_221220
'''

#%%
#Rawdata import(csv)
path = 'c:\\Users\\lenovo\\Desktop\\project'
#os.getcwd()
filepath_1 = os.path.join(path, "listings_220319.csv")
filepath_2 = os.path.join(path, "listings_220611.csv")
filepath_3 = os.path.join(path, "listings_220914.csv")
filepath_4 = os.path.join(path, "listings_221220.csv")

dt_2203 = pd.read_csv(filepath_1)
dt_2206 = pd.read_csv(filepath_2)
dt_2209 = pd.read_csv(filepath_3)
dt_2212 = pd.read_csv(filepath_4)

# %%
#Basic check
print(f'[Data 1] 22.3.19(dt_2203)\n')
print(f'A.Type: {type(dt_2203)}\n')
print(f'B.Shape: {dt_2203.shape}\n')
print(f'C.Features:\n{dt_2203.dtypes}\n')

#%%
print(f'[Data 2] 22.6.11(dt_2206)\n')
print(f'A.Type: {type(dt_2206)}\n')
print(f'B.Shape: {dt_2206.shape}\n')
print(f'C.Features:\n{dt_2206.dtypes}\n')

#%%
print(f'[Data 3] 22.9.14(dt_2209)\n')
print(f'A.Type: {type(dt_2209)}\n')
print(f'B.Shape: {dt_2209.shape}\n')
print(f'C.Features:\n{dt_2209.dtypes}\n')

#%%
print(f'[Data 4] 22.12.20(dt_2212)\n')
print(f'A.Type: {type(dt_2212)}\n')
print(f'B.Shape: {dt_2212.shape}\n')
print(f'C.Features:\n{dt_2212.dtypes}\n')

# %%
#%%
#Column check (verify with document)
#filepath = os.path.join(path, "columns_2203.csv")
#dt_2203.dtypes.to_csv(path)

dt_2203.dtypes.to_csv("columns_2203.csv")
dt_2206.dtypes.to_csv("columns_2206.csv")
dt_2209.dtypes.to_csv("columns_2209.csv")
dt_2212.dtypes.to_csv("columns_2212.csv")

#%%[markdown]
'''
#### `2. Data explore/cleansing`
- Initial columns: 51 (pk:'id','host_id' )
'''

# %%
#Initial columns: 50
feat = ['id','host_id','price','minimum_nights','maximum_nights','minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights','minimum_nights_avg_ntm','maximum_nights_avg_ntm','has_availability','instant_bookable','host_since','host_acceptance_rate','host_is_superhost','host_total_listings_count','host_verifications','host_has_profile_pic','license','calculated_host_listings_count','calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms','host_location','host_neighbourhood','neighbourhood_cleansed','availability_30','availability_60','availability_90','availability_365','number_of_reviews','number_of_reviews_ltm','number_of_reviews_l30d','first_review','last_review','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','reviews_per_month','description','property_type','room_type','accommodates','bathrooms_text','beds','amenities']

#%%
#String vars: Need to check 
feat_obj = ['price','has_availability','instant_bookable','host_since','host_acceptance_rate','host_is_superhost','host_verifications','host_has_profile_pic','license','host_location','host_neighbourhood','neighbourhood_cleansed','first_review','last_review','description','property_type','room_type','bathrooms_text','amenities']

# %%
df_2203 = dt_2203[feat]
df_2206 = dt_2206[feat]
df_2209 = dt_2209[feat]
df_2212 = dt_2212[feat]

#%%[markdown]
'''
##### Data 22.12 (listings_221220)
'''
#%%
cols = df_2212.columns.values.tolist()
len(cols)  #check
# %%
df_2212.describe(include='all') #If category
df_2212.describe(include='all').to_csv("Static_2212.csv")
#%%
#df_2203.describe(include='all').to_csv("Static_2203.csv")
#df_2206.describe(include='all').to_csv("Static_2206.csv")
#df_2209.describe(include='all').to_csv("Static_2209.csv")

# %%
#PK check: ID verified
temp = copy.deepcopy(df_2212)
temp = temp.sort_values(by='id')  #
temp[temp.id.duplicated()]    

#%%
#Null check, data unique value

#%%
#price- Need to delete $
df_2212.price.unique()  

# %%
#sample host review
df_2212.head(5)  #id 3686, host id 4645, $67
#%%
sam = df_2212[df_2212["id"] == 3686]
sam2 = df_2203[df_2203["id"] == 3686]
sam3 = df_2206[df_2206["id"] == 3686]
sam4 = df_2209[df_2209["id"] == 3686]

pd.concat([sam,sam2,sam3,sam4], axis=0)

#%%[markdown]
'''
#### `2. [Important]Define target variable `
- 
availability_365
number_of_reviews_l30d
review_scores_rating
'''
#%%
plt.hist(df_2212['price'],alpha=0.5, label='availability_365')

#%%
plt.hist(df_2212['availability_365'],alpha=0.5, label='availability_365')

#%%
plt.hist(df_2212['number_of_reviews_l30d'],alpha=0.5, label='number_of_reviews_l30d')
#%%
plt.hist(df_2212['review_scores_rating'],alpha=0.5, label='review_scores_rating')

#%%
df_2212['number_of_reviews_l30d'].value_counts()

#%%[markdown]
'''
#### `3. Feature analysis`
- 
'''
# %%
# Get one hot encoding of columns B
one_hot = pd.get_dummies(df['B'])

#%%[markdown]
'''
#### `Review data
- 
'''
#%%
#sourcing review data
filepath = os.path.join(path, "reviews_221220.csv")

rv_2212 = pd.read_csv(filepath)

# %%
rv_2212.head(3)

# %%
df_2212[df_2212["id"] == 3686].number_of_reviews

#%%
sam = rv_2212[rv_2212["listing_id"] == 3686]
sam