#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import copy
import os
import pickle #dataframe save and load
import openpyxl

#%%[markdown]
'''
## `Analysis for Air bnb data in D.C` <br>
### `- Team3`
 
#### **Raw data Introduction**
Air bnb Host data in washington dc (Sourcing from [Inside Airbnb](http://insideairbnb.com/data-assumptions "data assumption")

 The data utilizes public information compiled from the Airbnb web-site including the availabiity calendar for 365 days in the future, and the reviews for each listing. Data is verified, cleansed, analyzed and aggregated.
 
 There are four types of dataset in   
 1. Listings(Summary info of the property **on the Base date**) 
   - Property(Room info,Location), Host info, 
   - Deal info(Price, availability)
   - Reputation(# of reviews, avergae scores)
 
 2. Calendar: Calendar availablity that hosts set, used to define '(future)availability' in Listings
 
 3. Reviews: Customer review (date, score,comments)
 4. Neighbourhood: Categorical distirict groups to define used to 'neighbourhood_group' in Listings

We used "Listings" data as a basement, four datasets (=four base dates)
Jun 11,2022 /Sep 14, 2022/ Dec 20, 2022/ Mar 19,2023

Core Keywords for analysis
id, host_id
neighbourhood
superhost: experienced, highly rated hosts who are committed to providing great stays for guests. (Airbnb)
property_type
room_type
availability
review score

'''

#%%[markdown]
'''
#### `1. Data Import`
- Number of columns (raw data): 75 (pk:'id')
- dt_2206 ~ dt_2303 (four datasets)
'''

#%%
#Rawdata import(csv)
dt_2206 = pd.read_csv("listings_220611.csv")
dt_2209 = pd.read_csv("listings_220914.csv")
dt_2212 = pd.read_csv("listings_221220.csv")
dt_2303 = pd.read_csv("listings_230319.csv")

#path = 'c:\\Users\\lenovo\\Desktop\\Project_DM'
#filepath_1 = os.path.join(path, "listings_230319.csv")

#%%
print(f'[Data 1] 22.6.11(dt_2206)\n')
print(f'A.Type: {type(dt_2206)}\n')
print(f'B.Shape: {dt_2206.shape}\n')
print(f'C.Features:\n{dt_2206.dtypes}\n')

#%%
print(f'[Data 2] 22.9.14(dt_2209)\n')
print(f'A.Type: {type(dt_2209)}\n')
print(f'B.Shape: {dt_2209.shape}\n')
print(f'C.Features:\n{dt_2209.dtypes}\n')

#%%
print(f'[Data 3] 22.12.20(dt_2212)\n')
print(f'A.Type: {type(dt_2212)}\n')
print(f'B.Shape: {dt_2212.shape}\n')
print(f'C.Features:\n{dt_2212.dtypes}\n')

# %%
#Basic check
print(f'[Data 4] 23.3.19(dt_2303)\n')
print(f'A.Type: {type(dt_2303)}\n')
print(f'B.Shape: {dt_2303.shape}\n')
print(f'C.Features:\n{dt_2303.dtypes}\n')

#Column check (verify with document)
#dt_2203.dtypes.to_csv("columns_2203.csv")
#dt_2206.dtypes.to_csv("columns_2206.csv")
#dt_2209.dtypes.to_csv("columns_2209.csv")
#dt_2212.dtypes.to_csv("columns_2212.csv")

#%%
#PK check: ID verified
temp = copy.deepcopy(dt_2206)
temp = temp.sort_values(by='id')  #
temp[temp.id.duplicated()]    

#%%[markdown]
'''
#### `2-1. Data explore(pre-EDA)`
'''

#%%
print(dt_2206['host_response_time'].unique())
print(dt_2206['host_response_rate'].unique())
print(dt_2206['host_acceptance_rate'].unique())
print(dt_2206['neighbourhood_cleansed'].unique())
print(dt_2206['neighbourhood'].unique())
print(dt_2206['property_type'].unique())
print(dt_2206['room_type'].unique())
print(dt_2206['accommodates'].unique())
print(dt_2206['bedrooms'].unique())
print(dt_2206['beds'].unique())
print(dt_2206['license'].unique())

#%%
dt_2206.host_response_time.value_counts(dropna=False)
dt_2206.host_acceptance_rate.value_counts(dropna=False)
dt_2206.host_is_superhost.value_counts(dropna=False)

dt_2206["calculated_host_listings_count"].value_counts(normalize=True).sort_index().mul(100).round(1)

#%%
#print(dt_2206['license'].unique())  #Categorical values
#%%
#dt_2206.has_availability.value_counts(dropna=False).sort_index() #Freq
#%%
#dt_2206["calculated_host_listings_count"].value_counts(normalize=True).sort_index().mul(100).round(1) #Freq(Ratio)

#%%
pd.crosstab(dt_2206.has_availability,dt_2206.number_of_reviews_l30d, normalize='index').mul(100).round(1)  #Pivot Freq(Ratio)

#%%[markdown]
'''
#### `Target to predict(draft- will continue at EDA)`
- Availability/Occupancy (has_availability, availability_30,60,90,365 )
- Recent booking: Number of review in a month(number_of_reviews_l30d, 365) 
- Number of reviews: total, in ltm, in a month 

[Case review]
a. has_availability=f (availability=0, instant_bookable=f, number_of_reviews=0 
: Biz stopped

b. has_availability=t & instant_bookable = t 
& availability_n high(low occupancy) & number_of_reviews=0 
: Bad performance or New hosts[by Biz years?]

'''

#%%
#has_availability
dt_2206["has_availability"].value_counts(normalize=True).sort_index().mul(100).round(1)
#dt_2206.has_availability.value_counts(dropna=False).sort_index()

#%%
#has_availability x instant_bookable
pd.crosstab(dt_2206.has_availability,dt_2206.instant_bookable, normalize='index').mul(100).round(1)

#%%
#has_availability x availability_30
pd.crosstab(dt_2206.has_availability,dt_2206.availability_30, normalize='index').mul(100).round(1)

#%%
#has_availability x availability_365
pd.crosstab(dt_2206.has_availability,dt_2206.availability_365, normalize='index').mul(100).round(1)

#%%
#has_availability x number_of_reviews_l30d
pd.crosstab(dt_2206.has_availability,dt_2206.number_of_reviews_l30d, normalize='index').mul(100).round(1)

#%%
#has_availability x number_of_reviews_ltm
pd.crosstab(dt_2206.has_availability,dt_2206.number_of_reviews_ltm, normalize='index').mul(100).round(1)

#%%
pd.crosstab(dt_2206.has_availability,dt_2206.availability_30, normalize='index').mul(100).round(1)

#%%
pd.crosstab(dt_2206.availability_30,dt_2206.number_of_reviews_l30d, normalize='index').mul(100).round(1)

#%%
'''s = dt_2206[dt_2206["availability_30"] == 0]
s.has_availability.value_counts()
pd.crosstab(s.has_availability,s.instant_bookable, normalize='index').mul(100).round(1)'''

# %%
#
print("48 columns have been picked for analysis")

feat = ['id','minimum_nights','availability_30','availability_60','availability_90','availability_365','price','has_availability','instant_bookable','host_id','host_total_listings_count','calculated_host_listings_count','calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms','host_name','host_since','host_acceptance_rate','host_response_time','host_is_superhost','host_neighbourhood','host_verifications','host_has_profile_pic','host_identity_verified','license','neighbourhood','neighbourhood_cleansed','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','reviews_per_month','number_of_reviews','number_of_reviews_ltm','number_of_reviews_l30d','first_review','last_review','bedrooms','beds','accommodates','name','property_type','room_type','bathrooms_text','amenities']

#String vars: Need to check 
feat_obj = ['price','has_availability','instant_bookable','host_name','host_since','host_acceptance_rate','host_response_time','host_is_superhost','host_neighbourhood','host_verifications','host_has_profile_pic','host_identity_verified','license','neighbourhood','neighbourhood_cleansed','first_review','last_review','name','property_type','room_type','bathrooms_text','amenities']


#%%[markdown]
'''
#### `2-2. Data wrangling(pre-EDA)`
- df_2206 ~ df_2303 (48 columns have been picked for analysis)
- Wrong records exclude(dt_2209,dt_2303)
- Merge to one dataset ('date' feat add, will split again when modeling)
- Change Object to int (replace)

'''

# %%

df_2206 = dt_2206[feat]
df_2209 = dt_2209[feat]
df_2212 = dt_2212[feat]
df_2303 = dt_2303[feat]

#%%
#Make date column before merge
df_2206['date'] = "2206"
df_2209['date'] = "2209"
df_2212['date'] = "2212"
df_2303['date'] = "2303"

print("Move on to the next step")
#df_2206.insert(2, 'date', '2206', True)

#%%[markdown]
'''
#####  Merge to one dataset 
- Check if wrong records exist
'''

#%%
print("3.Merge: 4 datasets")
df = pd.concat([df_2206,df_2209,df_2212,df_2303], axis=0).reset_index(drop=True) 
# %%
print(f'After merge: {df.shape}')
#df.tail(5)
print(f'Check sum of 4 datasets: {df_2206.shape[0] + df_2209.shape[0] + df_2212.shape[0] + df_2303.shape[0]}')

#%%
print("Move on to the next step")

#%%
'''
#Skip
print("If you found another wrong records- keep cleansing")
print("a. host_has_profile_pic: delete 1 records")
print(f'Before cleansing: {df.shape}')
filter = df["host_has_profile_pic"] != ' private entrance'
df = df[filter]
print(f'After cleansing: {df.shape}')
df = df.reset_index(drop=True) #important!
df.tail(5)
'''

#%%
print("Save to current dataframe to save as pickle(temp)")
#Save Merged dataset(before replace)- Apr 9
#df.to_pickle("./df_init.pkl")
#df = pd.read_pickle("./df_init.pkl")

#%%[markdown]
'''
##### Change Object to int (replace)
- Replace boolean(T/F) into 1/0 > 4 features (can be added more after EDA)
has_availability_2 : t/f - 1/0
host_has_profile_pic: f,t,nan
host_is_superhost: f,t,nan
host_identity_verified: f,t,nan

- Re-define Categories or converting(if necessary)
Price, License(TBD): 
'''

# %%
def replace_bool(df,var,var_2):
    '''
    function for boolean category: True/False/Nan > 1/0/Nan
    df, var(orginal feature),var_2(New feature replaced)
    '''
    print(f'Before Replacing: {df.shape}\n')
    print(f'Check NaN:\n {df[var].value_counts(dropna=False).sort_index()}')
    #Replace value: string to int 
    df[var_2] = df[var].map(lambda x: 1 if x=='t'else 0 if x=='f' else np.nan)
    print(f'\nAfter Replacing(add 1 var): {df.shape}')
    print(f'\nVerify result\n: {pd.crosstab(df[var],df[var_2],margins=True)}')
    print(f'\nVerfy NaN:\n {df[var_2].value_counts(dropna=False).sort_index()}')
    return #check

#%%

print("4.Replace boolean(T/F) into 1/0- 4 features")
'''has_availability_2 : t/f - 1/0
host_has_profile_pic: f,t,nan
host_is_superhost: f,t,nan
host_identity_verified: f,t,nan'''

#%%
replace_bool(df,'has_availability','has_availability_2')
#%%
replace_bool(df,'host_has_profile_pic','host_has_profile_pic_2')

#%%
replace_bool(df,'host_is_superhost','host_is_superhost_2')

#%%
replace_bool(df,'host_identity_verified','host_identity_verified_2')

#%%
print("Save to current dataframe to save as pickle(temp)")
#Save Merged dataset(before replace)- Apr 9
#df.to_pickle("./df_init.pkl")
#df = pd.read_pickle("./df_init.pkl")

#%%
print("Move on to the next step")

#%%[markdown]
'''
#### `2-3. EDA(TBD)`
- Drawing charts and tables
- Single, Pivot (between preditors or preditor x target)
- Feature wrangling(add)
- Re-catagorize, Fill n/a (prepare to dummify)
- Feature engineering 
'''

#%%
print("Check current status again")

print(df.shape)
print(df.date.value_counts().sort_index())

#%%
print("Stat(describe)")

# %%
col = df.columns.values.tolist()
len(col) 
df.dtypes.to_csv("Cols_2.csv")

#%%
df.describe(include='all').to_csv("EDA_Static.csv")


#%%
print("############### 4/10: Will continue from this line ############")

#%%
print("Please load pickle file below")
df = pd.read_pickle("./df_init.pkl")
#df.to_pickle("./df_init.pkl")

#%%
print("Check Price featrue(TBD)")

#%%
#price- Need to delete $
df.price.unique()  

# %%
#sample host review
df.head(5)  #id 3686, host id 4645, $67
#%%
sam = df[df["id"] == 3686]

#%%[markdown]
'''
#### `Target to predict(Con'd, TBD)`
- 1.Availability/Occupancy (has_availability, availability_30,60,90,365 )
- 2.Recent booking: Number of review in a month(number_of_reviews_l30d, 365) 
- 3.Number of reviews: total, in ltm, in a month 
- 4.Price, review_scores_rating

[Case review]
a. has_availability=f (availability=0, instant_bookable=f, number_of_reviews=0 
: Biz stopped

b. has_availability=t & instant_bookable = t 
& availability_n high(low occupancy) & number_of_reviews=0 
: Bad performance or New hosts[by Biz years?]

'''

#%%
plt.hist(df['price'],alpha=0.5, label='availability_365')

#%%
plt.hist(df['availability_365'],alpha=0.5, label='availability_365')

#%%
plt.hist(df['number_of_reviews_l30d'],alpha=0.5, label='number_of_reviews_l30d')
#%%
plt.hist(df['review_scores_rating'],alpha=0.5, label='review_scores_rating')

#%%
df_2212['number_of_reviews_l30d'].value_counts()

#%%
print("Check Other featrue(TBD)")

#%%[markdown]
'''
#### `3. Feature Engineering(TBD)`
- Get Dummy
'''
# %%
# Get one hot encoding of columns B
#one_hot = pd.get_dummies(df['B'])

#%%[markdown]
'''
#### `3. Feature Engineering(TBD)`
- Text mining (review comments)
'''
#%%
#sourcing review data
#filepath = os.path.join(path, "reviews_221220.csv")
#rv_2212 = pd.read_csv(filepath)
#df_2212[df_2212["id"] == 3686].number_of_reviews
#sam = rv_2212[rv_2212["listing_id"] == 3686]
