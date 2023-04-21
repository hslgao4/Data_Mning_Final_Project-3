# %%
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

#%%[markdown]
'''
## `Analysis for Air bnb data in D.C` <br>
### `- Team3`
 
#### **Raw data Introduction**
Air bnb Host data in washington dc (Sourcing from [Inside Airbnb](http://insideairbnb.com/data-assumptions "data assumption")

 The data utilizes public information compiled from the Airbnb web-site including the availabiity calendar for 365 days in the future, and the reviews for each listing. 
 
The dataset has information about    
   - Listings(Room info, Location)
   - Host info(Number of host years, number of properties that host has)
   - Deal info(Price, availability)
   - Reputation(# of reviews, average review scores)
 
We used "Listings.csv" data on Jun 11,2022, which has 6308 listings at the date.

Keywords for analysis
id(Listings), host_id(Host)
superhost: experienced, highly rated hosts who are committed to providing great stays for guests. (Airbnb)
availability: reflect hosts' plan and pre-bookings of customer. 
review_scores_rating: average score of past reviews (6 specific sectors)
review_scores_accuracy, review_scores_cleanliness, review_scores_checkin
review_scores_communication, review_scores_location, review_scores_value

'''

#%%[markdown]
'''
#### `1. Data Import`
- Number of records: 6308 (pk:'id')
'''

#%%
df = pd.read_csv("listings_220611.csv")
df.shape


#%%[markdown]
#Temporary
#- Alive two cols (I identified relatoinship between these two and review score when seeing confusion matrix"

#%%
#Change type of key (int)
#df.head(5)
df.id = df.id.astype(int)
#df.dtypes

#%%
#Import previous rawdata (containing all cols)
dt_2206 = pd.read_csv("listings_220611_init.csv")
#dt_2209 = pd.read_csv("listings_220914.csv")
#dt_2303 = pd.read_csv("listings_230319.csv")

#%%
dt_2206.head(5)

#%%
'''filter = dt_2206.host_id == 107434423
temp = dt_2206[filter]
temp'''

#%%
df.head(5)

#%%
'''filter = df.host_id == 107434423
temp = df[filter]
temp
'''
#%%
#rename PK for join to the initial
dt_2206.rename(columns={'id': 'listing_id'},inplace=True)
#%%
#filter = dt_2206.host_id == 55615870
#temp = dt_2206[filter]
#temp
#%%
temp = copy.deepcopy(df)

# %%
# 
col = ['listing_id','host_response_time','host_is_superhost']
temp2 = dt_2206[col]

#%%
#Join with index: 
temp = temp.join(temp2)
#Join with the key


#%%
temp.shape
#%%
#temp.shape
temp.tail(5)
#df.shape
#%%
df = temp
#%%[markdown]
'''
#### `2. Define binary target (review_scores_rating_t)`
- 0: If review_scores_rating >= 4.5 (High) 
- 1: If review_scores_rating < 4.5 (Low)
- -9: review_scores_rating = NaN (1349, 21.4%),  no review score (number_of_reviews=0 )
      *temporary, convert NaN into -9 (make a decision later)

- Made two version 
  ver1. review_scores_rating_t (at the moment of evaluation: Jun 11,2022)
  ver2. review_scores_rating_t2 (after 6M, at the moment of Dec 20,2022)
  
Dataset: make model (0,1) - train 70%,test 30%) 
  
'''

#%%
df['review_scores_rating_t'] = df['review_scores_rating'].apply(lambda x: 0 if x >= 4.5 else 1 if x < 4.5 else -9)
#Replace NaN -> -9
#%%
df.review_scores_rating_t.value_counts(dropna=False)
#%%
df.review_scores_rating_t.value_counts(dropna=False,normalize=True).sort_index().mul(100).round(1)

#%%[markdown]
#
'''
Make target on Mar 2023 (review_scores_rating_t2 )
 review_scores_rating_t2(after 6M, at the moment of Dec 20,2022) Will be joined to the listing of the df (Jun 2022) 
'''
#%%
#Import previous rawdata (containing all cols)
dt_2303 = pd.read_csv("listings_230319.csv")
dt_2303.rename(columns={'id': 'listing_id'},inplace=True) #change pk name
#%%
dt_2303['review_scores_rating_t2'] = dt_2303['review_scores_rating'].apply(lambda x: 0 if x >= 4.5 else 1 if x < 4.5 else -9)

#dt_2303.loc[dt_2303['review_scores_rating'] >= 4.5, 'review_scores_rating_t2'] = 0
#dt_2303.loc[dt_2303['review_scores_rating'] < 4.5, 'review_scores_rating_t2'] = 1
#dt_2303.review_scores_rating_t2.fillna(-9, inplace = True)

#%%
dt_2303.review_scores_rating_t2.value_counts(dropna=False,normalize=True).sort_index().mul(100).round(1)

#%%
#Left Join review_scores_rating_t2 (review after 6M)
col = ['listing_id','review_scores_rating_t2']
temp2 = dt_2303[col]
df = pd.merge(df, temp2, how='left', on=['listing_id'])
df.tail(5)

#%%
pd.crosstab(df['review_scores_rating_t'],df['review_scores_rating_t2'],dropna=False, margins=True, margins_name="Total")

#%%
#Adjust target2 is nan(no listing) or -9(no review) by assumption
temp = copy.deepcopy(df)

#%%
#Will change to -9
filter1 = (temp.review_scores_rating_t2.isna()==1) & (temp.review_scores_rating_t ==-9)
#Will change to review_scores_rating_t
filter2= ((temp.review_scores_rating_t2.isna()==1) & (temp.review_scores_rating_t !=-9))|((temp.review_scores_rating_t2==-9) & (temp.review_scores_rating_t !=-9))

#filter3= (temp.review_scores_rating_t2==0)|(temp.review_scores_rating_t2==1)|((temp.review_scores_rating_t==-9)&(temp.review_scores_rating_t2==-9))

#%%
#assign
temp.loc[filter1, 'review_scores_rating_t2'] = -9
temp.loc[filter2, 'review_scores_rating_t2'] = temp.loc[filter2, 'review_scores_rating_t']
#temp3 = temp.loc[filter3, :] #no change
temp.shape
#temp.tail(5)

#%%
pd.crosstab(temp['review_scores_rating_t'],temp['review_scores_rating_t2'],dropna=False, margins=True, margins_name="Total")


#%%
df = temp
print("End of the target definition. Move on to the next step")

#########################################################

#%%[markdown]
'''
#### `3. Data Explore
- EDA, transform, prepare to modeling,etc. 
'''

#%%[markdown]
#14 Apr, repeat Liangs's job
# %%
df['price_per_room'] = df['price'] / df['bedrooms']
df.shape
#df.dtypes

#%%
'''
Transform: Boolean(T/F) -> Int (1/0)
'''
def replace_bool(df,var):
    df[var] = df[var].map(lambda x: 1 if x=='t'else 0 if x=='f' else np.nan)

replace_bool(df,'has_availability')
replace_bool(df,'host_has_profile_pic')
replace_bool(df,'host_identity_verified')  
replace_bool(df,'host_is_superhost')  
df.shape  

# %%
'''
Transform: Rename
'''
df = df.rename(columns={'host_acceptance_rate': 'host_accept.R', 'host_listings_count': 'Listings.DC', 'host_total_listings_count': 'Listings.Total', 'host_has_profile_pic': 'profile.pic',
                        'host_identity_verified': 'identity.verify', 'minimum_nights': 'min_nights', 'maximum_nights': 'max_nights', 'has_availability':'availability', 
                         'availability_30': 'avail_30', 'availability_60': 'avail_60', 'availability_90':'avail_90', 'availability_365': 'avail_365'
                             })
#df.columns

#%%
#%%
print("Save to current dataframe to save as pickle(temp)")
#Save 22.6 dataset
#df.to_pickle("./df_clean_v2.pkl")
df = pd.read_pickle("./df_clean_v2.pkl") #4/15

#%%
df.head(5)

#%%[markdown]
'''
#### `3. [Add features] Review text mining(22.6)
# of total reviews: 295978
# of unique listings: 4958
'''
#%%
#%%
rv = pd.read_csv("reviews_220611.csv")
rv.shape
# %%
rv.dtypes
#%%
#PK check
rv = rv.sort_values(by='id') 
rv[rv.id.duplicated()] #no dup

# %%
#cleansing null
rv = rv.dropna()
# %%
rv.shape

# %%
#check # of unique listings that have review
temp = copy.deepcopy(rv)
temp = temp.sort_values(by='listing_id')  #
temp = temp.drop_duplicates(subset=['listing_id'])     #delete dup, sort/dup check 
# %%
print(f'number of unique listings: {temp.shape[0]}')

# %%
import nltk
#%%
nltk.download('vader_lexicon')
# %%
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
# %%
'''
Apply built-in analyzer in the NLTK Python library 
- Assign score of strength(polarity) 
 whether a comment is 'positive'or 'negative' or 'neutral'
'''
rv['polarity_value']="Default"
rv['neg']=0.0
rv['pos']=0.0
rv['neu']=0.0
rv['compound']=0.0
for index,row in rv.iterrows():
    ss = sid.polarity_scores(row['comments'])
    rv.at[index,'polarity_value'] = ss
    rv.at[index,'neg'] = ss['neg']
    rv.at[index,'pos'] = ss['pos']
    rv.at[index,'neu']= ss['neu']
    rv.at[index,'compound'] = ss['compound']
rv.head()
# %%
from langdetect import detect
def detect_lang(sente):
    sente=str(sente)
    try:
        return detect(sente)
    except:
        return "None"

for index,row in rv.iterrows():
    lang=detect_lang(row['comments'])
    rv.at[index,'language'] = lang
#     print(lang)
#%%
print(rv.language.value_counts(dropna=False,normalize=True))

#Save Merged dataset(before replace)- Apr 9
#df.to_pickle("./df_init.pkl")
#df = pd.read_pickle("./df_init.pkl")

#%%
#taking rows whose language is English
rv_eng=rv[rv.language=='en']

#%%
'''
Frequency
'''
polarDF=rv_eng[['pos']]
polarDF=polarDF.groupby(pd.cut(polarDF["pos"], np.arange(0, 1.1, 0.1))).count()
polarDF=polarDF.rename(columns={'pos':'count_of_Comments'})
polarDF=polarDF.reset_index()
polarDF=polarDF.rename(columns={'pos':'range_i'})
for i,r in polarDF.iterrows():
    polarDF.at[i,'RANGE'] = float(str(r['range_i'])[1:4].replace(',',''))
    polarDF.at[i,'Sentiment'] = 'positive'
del polarDF['range_i']
polarDF.head()

#%%
polarDFneg=rv_eng[['neg']]
polarDFneg=polarDFneg.groupby(pd.cut(polarDFneg["neg"], np.arange(0, 1.1, 0.1))).count()
polarDFneg=polarDFneg.rename(columns={'neg':'count_of_Comments'})
polarDFneg=polarDFneg.reset_index()
polarDFneg=polarDFneg.rename(columns={'neg':'range_i'})
for i,r in polarDFneg.iterrows():
    polarDFneg.at[i,'RANGE'] = float(str(r['range_i'])[1:4].replace(',',''))
    polarDFneg.at[i,'Sentiment'] = 'negative'
del polarDFneg['range_i']
for i,r in polarDFneg.iterrows():
    polarDF=polarDF.append(pd.Series([r[0],r[1],r[2]],index=['count_of_Comments','RANGE','Sentiment']),ignore_index=True)
    
polarDFneg.head()

#%%
polarDFneut=rv_eng[['neu']]
polarDFneut=polarDFneut.groupby(pd.cut(polarDFneut["neu"], np.arange(0, 1.0, 0.1))).count()
polarDFneut=polarDFneut.rename(columns={'neu':'count_of_Comments'})
polarDFneut=polarDFneut.reset_index()
polarDFneut=polarDFneut.rename(columns={'neu':'range_i'})
for i,r in polarDFneut.iterrows():
    polarDFneut.at[i,'RANGE'] = float(str(r['range_i'])[1:4].replace(',',''))
    polarDFneut.at[i,'Sentiment'] = 'neutral' 
del polarDFneut['range_i']

for i,r in polarDFneut.iterrows():
    polarDF=polarDF.append(pd.Series([r[0],r[1],r[2]],index=['count_of_Comments','RANGE','Sentiment']),ignore_index=True)
    
polarDFneut.head()

#%%
import seaborn as sns
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
#plt.figure(figsize=(10,10))
sns.scatterplot(x = polarDF['RANGE'], y = polarDF['count_of_Comments'],hue=polarDF['Sentiment']) 

#%%
print("Save to current dataframe to save as pickle(temp)")
#Save 22.6 dataset
#polarDF.to_pickle("./review_sum.pkl")
#rv.to_pickle("./review.pkl")
#rv_eng.to_pickle("./review_eng.pkl")

polarDF = pd.read_pickle("./review_sum.pkl")
rv = pd.read_pickle("./review.pkl")
rv_eng = pd.read_pickle("./review_eng.pkl")
#df = pd.read_pickle("./df_clean_v2.pkl") #4/15
rv.dtypes

#%%
col = rv.columns.values.tolist()
col
#%%
print("End of the 3. [Add features] Review text mining")

###################################################

#%%
print("Load current dataframe and join review features")
#df = pd.read_pickle("./df_clean_v2.pkl") #4/15

#%%
df.shape

#%%
#%%
rv.head(5)

#%%
cols = ['listing_id','neg','pos','neu']
temp = rv[cols]
temp.head(5)
#%%
'''
Make an summary statstics of comment keyword score (Negative, Positve, Neutral) per the property
 - Mean: Average of Negative,Postive,Neutral score (of reviews) in each property 
 - Std: Variation of Negative,Postive,Neutral score (of reviews) in each property 
'''

neg = temp.groupby('listing_id')["neg"].mean().round(3).reset_index(name='Avg_neg_review_comment_score')
neg.head(5)

#%%
neg_dev = temp.groupby('listing_id')["neg"].std().round(3).reset_index(name='Std_neg_review_comment_score')
neg_dev.head(5)

#%%
pos = temp.groupby('listing_id')["pos"].mean().round(3).reset_index(name='Avg_pos_review_comment_score')
pos.head(5)

#%%
pos_dev = temp.groupby('listing_id')["pos"].std().round(3).reset_index(name='Std_pos_review_comment_score')
pos_dev.head(5)

#%%
neu = temp.groupby('listing_id')["neu"].mean().round(3).reset_index(name='Avg_neu_review_comment_score')
neu.head(5)

#%%
neu_dev = temp.groupby('listing_id')["neu"].std().round(3).reset_index(name='Std_neu_review_comment_score')
neu_dev.head(5)

#%%
'''
Join new features (review comment) to the original dataset
'''
#%%
print(temp.shape)
temp = pd.merge(temp, neu, how='left', on=['listing_id'])
print(temp.shape)

#%%
print(temp.shape)
temp = pd.merge(temp, neu, how='left', on=['listing_id'])
print(temp.shape)

#%%
print(temp.shape)
temp = pd.merge(temp, neu, how='left', on=['listing_id'])
print(temp.shape)

#%%
df = temp
#df.shape
temp = copy.deepcopy(df)

#%%
print(df.shape)
temp = pd.merge(df, neg_dev, how='left', on=['listing_id'])
print(temp.shape)


#%%
print(temp.shape)
temp = pd.merge(temp, pos_dev, how='left', on=['listing_id'])
print(temp.shape)

#%%
print(temp.shape)
temp = pd.merge(temp, neu_dev, how='left', on=['listing_id'])
print(temp.shape)

#%%
df = temp
df.shape

#%%
df.tail(5)
#%%
df.dtypes
#%%
'''
Join new features 2: number of amenities
'''
#%%
temp = copy.deepcopy(df)

#%%
#just sourcing from the original dataset
col = ['listing_id','amenities']
temp2 = dt_2206[col]

#%%
#Join with the key
print(temp.shape)
temp = pd.merge(temp, temp2, how='left', on=['listing_id'])
print(temp.shape)

#%%
#temp.dtypes
print(type(temp['amenities']))

#%%
'''
amenities is series, so to find the number of list elements, change type to list by split string 
'''
temp.amenities = temp.amenities.str.split(',') 
temp['num_amenities'] = temp.amenities.apply(lambda x: len(x))

col = ['id','amenities','num_amenities']
temp[col].tail(5)
#%%
df = temp


#%%
'''
Join new features 3: number of biz
'''
temp = copy.deepcopy(df)
#%%
from datetime import datetime

# Convert the business start date to a datetime object
temp['business_start_date'] = pd.to_datetime(temp['host_since'], format='%m/%d/%y')
temp.tail(5)
#%%
# Calculate the number of years in business
current_date = pd.to_datetime('6/11/22', format='%m/%d/%y')
temp['years_in_business'] = (current_date - temp['business_start_date']).dt.days / 365.25

temp.tail(5)
#%%
df = temp

#%%
'''
Replace boolean 
'''
df.instant_bookable.value_counts()

#%%
replace_bool(df,'instant_bookable')  
df.instant_bookable.value_counts()
#%%
df.dtypes

#%%
#df.head(5)

#%%
'''
Reviewing number of listings vars
calculated_host_listings_count is most reliable
'''
#df[df['Listings.DC'] == 0].shape[0]
#df[df['Listings.DC'] == 0].head(5)
#df[df['host_id'] == 33736673]
#df[df['calculated_host_listings_count'] == 0].shape[0]
#df[df['calculated_host_listings_count'] == 2].head(5)


###############################################
#%%[markdown]
'''
#### `4. Predictive modeling
Step 1. Final wrangling before modeling
Step 2. EDA, Examine Correlation candidate input vars w target vars
Step 3. Train, Test dataset split (Hold-out or Cross validatoin)
Step 4. Fit the models using classficiation algorithms
Step 5. Evaluate models  
'''

#%%[markdown]
'''
Step 1. Final wrangling before modeling

Two feature group list
num = [Int or float]
cat = [objects] > get_dummy of each categoreis

'''

#%%
cat = ['room_type','host_response_time']

#%%
def make_dummies(df, cols):
    return pd.get_dummies(df, columns=cols)

#Skipped label encoding
# Step 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
#le = preprocessing.LabelEncoder()
# Step 2. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
#X_2 = X.apply(le.fit_transform)
#X_2.head()

# %%
df = make_dummies(df, cat)

# %%
df.dtypes

#%%
print("Save to current dataframe to save as pickle(temp)")

#df.to_pickle("./df_clean_min.pkl")
df = pd.read_pickle("./df_clean_min.pkl") #4/15

# %%
df.shape
#%%

print("This is final features that would be used for modeling: 37 Input variables")
print("- Include one target variable and Primary Key")

feat = ['listing_id','host_accept.R','profile.pic','identity.verify','bedrooms','beds','reviews_per_month','price_per_room','Avg_neg_review_comment_score','Avg_pos_review_comment_score','Avg_neu_review_comment_score','Std_neg_review_comment_score','Std_pos_review_comment_score','Std_neu_review_comment_score','years_in_business','accommodates','price','min_nights','max_nights','availability','avail_60','number_of_reviews','number_of_reviews_l30d','instant_bookable','calculated_host_listings_count','calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms','num_amenities','room_type_Entire home/apt','room_type_Hotel room','room_type_Private room','room_type_Shared room','host_response_time_a few days or more','host_response_time_within a day','host_response_time_within a few hours','host_response_time_within an hour','review_scores_rating_t2','review_scores_rating_t']


#%%
df = df[feat]

#%%
#Choose target 0,1 (delete there is no review record)

filter = df.review_scores_rating_t2 != -9 #review_scores_rating_t
df2 = df[filter]
print(df2.shape)
#%%
print(df2.review_scores_rating_t2.value_counts())

#%%
print(f'Null value check:{df2.isnull().sum()}')

'''
#Below features have NaN value
Std_neg_review_comment_score 
Std_pos_review_comment_score
Std_neu_review_comment_score

Avg_pos_review_comment_score
Avg_neg_review_comment_score
Avg_neu_review_comment_score 

price_per_room
beds
bedrooms
host_accept.R 
'''

#%%
print(f'a. bedrooms=nan -> replace to 1')
#df[df['bedrooms'].isna()==1].beds.value_counts()
df2['bedrooms'].fillna(1, inplace = True)
df2['bedrooms'].value_counts()

#%%
df2['price_per_room'] = df2['price'] / df2['bedrooms']
df2['price_per_room'].value_counts()

#%%
print(f'Null value check:{df2.isnull().sum()}')
# %%
print(f'b. beds=nan -> replace to 0')
df2[df2['beds'].isna()==1].bedrooms.value_counts()
#%%
df2['beds'].value_counts()
#%%
df2['beds'].fillna(0, inplace = True)
df2['beds'].value_counts()

#%%
print(f'Null value check:{df2.isnull().sum()}')
#%%
print(f'c. host_accept.R=nan -> replace to Average: 89.6%')

df2['host_accept.R'].value_counts()

# %%
#85: accept%= 0
df2[df2['host_accept.R']==0].shape[0]
#%%
#Most of them is there is no review(booking)
df[df['host_accept.R'].isna()==1].number_of_reviews.value_counts()

#%%
#print(df2['host_accept.R'].mean())
df2['host_accept.R'].fillna(df2['host_accept.R'].mean(), inplace = True)
df2['host_accept.R'].value_counts()

#%%
print(f'Null value check:{df2.isnull().sum()}')

#%%
print(f'c. Std_neg_review_comment_score=nan -> replace to Average: ')

 
#%%
#Std=Na, just one review.
df2[df2['Std_neg_review_comment_score'].isna()==1].number_of_reviews.value_counts()
#%%
col = ['Std_neg_review_comment_score','Std_pos_review_comment_score','Std_neu_review_comment_score','Avg_pos_review_comment_score','Avg_neg_review_comment_score','Avg_neu_review_comment_score']

for i in col:
    df2[i].fillna(df2[i].mean(), inplace = True)

print(f'Null value check:{df2.isnull().sum()}')

#%%
print("Save to current dataframe to save as pickle(temp)")

#df2.to_pickle("./df_clean_min_fin.pkl")
df2 = pd.read_pickle("./df_clean_min_fin.pkl") #4/15

#%%[markdown]
'''
Step 2. EDA, Examine Correlation candidate input vars w target vars

'''

#%%
print("*1. Define functions drawing table/charts with vars")

def my_multi_vars(df, var1, var2):
    '''
    p1: freq, p2: ratio(%)
    '''
    p1 = pd.crosstab(df[var1],df[var2], margins=True, margins_name="Total")
    p2 = pd.crosstab(df[var1],df[var2], normalize='index').mul(100).round(1)
    print(f'A.Pivot table {var1} & {var2}(#)\n{p1}\n\nB.Pivot table {var1} & {var2}(%)\n{p2}') 

#%%
my_multi_vars(df2, 'beds', 'review_scores_rating_t2')

#%%

print("*2. Define functions drawing histogram with var and target")

def my_histchart(df, var, target):
    #print(f'Freq of residents by {var}')
    
    d1 = df[df[target] == 0]
    plt.subplot(1,2,1) 
    plt.hist(d1[var],alpha=0.5,color='blue',linewidth=1.2, edgecolor='black')
    plt.xlabel(var)
    plt.ylabel("Freq")
    plt.title(f'a.{var} for High review scores') 

    d2 = df[df[target] == 1]
    plt.subplot(1,2,2) 
    plt.hist(d2[var],alpha=0.5,color='orange',linewidth=1.2, edgecolor='black')
    plt.xlabel(var)
    plt.yticks([])
    plt.ylabel("Freq")
    plt.title(f'b.{var} for Low') 
    
    fig = plt.figure()
    plt.hist(d1[var],alpha=0.5,color='blue',linewidth=1.2, edgecolor='black',label='High')
    plt.hist(d2[var],alpha=0.5,color='orange',linewidth=1.2, edgecolor='black',label='Low')
    plt.legend(loc='upper right')
    plt.title(f'B.Histogram Overlap (High+ Low)')

# %%
my_histchart(df2, 'num_amenities', 'review_scores_rating_t2')

#%%
my_histchart(df2, 'price', 'review_scores_rating_t2')


#%%

print("*3. Define functions drawing Scatter chart with two vars and target")

def my_schart(df, var, var2, target):
    d1 = df[df[target] == 0]
    plt.subplot(1,2,1) 
    sns.scatterplot(x = d1[var], y = d1[var2],hue=d1[target]) 
    plt.title(f'{var} and {var2}-High(Blue)/Low(Red)') 
    
    d2 = df[df[target] == 1]
    plt.subplot(1,2,2) 
    sns.scatterplot(x = d2[var], y = d2[var2],hue=d2[var]) 
    #plt.title(f'{var} and {var2}-Low') 
    plt.yticks([])
    plt.show()


#%%
my_schart(df2, 'number_of_reviews', 'price', 'review_scores_rating_t2')


#%%[markdown]
'''
Step 3. Train, Test dataset split (Hold-out or Cross validatoin)

'''
#%%
print("Load final data again")
df2 = pd.read_pickle("./df_clean_min_fin.pkl") #4/15

#%%
print("Define input value)")
X = df2.drop(['review_scores_rating_t2','listing_id'], axis=1) #All input

print("Put chosen input values here)")
ip = ['identity.verify','beds','price_per_room','Avg_neg_review_comment_score','Avg_pos_review_comment_score','Avg_neu_review_comment_score','Std_neg_review_comment_score','Std_pos_review_comment_score','Std_neu_review_comment_score','years_in_business','price','min_nights','avail_60','number_of_reviews','calculated_host_listings_count']

#%%
X = X[ip]
y = df2['review_scores_rating_t2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
print(f'1.Low review customer(%)_Train: {round(sum(y_train)/X_train.shape[0]*100,2)}')
print(f'2.Low review customer(%)_Test:{round(sum(y_test)/X_test.shape[0]*100,2)}')
# %%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import xgboost

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

#%%


#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def fit_logistic_regression(X_train, y_train, X_test, y_test, cutoff, model_num):
    # Fit logistic regression model on training data
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, y_train)

    # Save the model
    model_name = 'logistic_regression_model.joblib'
    
    pickle.dump(lr, open(model_num, "wb"))
    
    # Print model coefficients and p-values
    print("Model Coefficients:")
    print(lr.coef_)
    print("\nModel Intercept:")
    print(lr.intercept_)
    #print("\nModel p-values:")
    #print(lr.pvalues_)

    # Evaluate the model on the test set
    y_pred_prob = lr.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= cutoff).astype(int)
    
    # Calculate evaluation metrics
    auroc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr = (y_pred == 1)[y_test == 0].sum() / (y_test == 0).sum()
    fnr = (y_pred == 0)[y_test == 1].sum() / (y_test == 1).sum()

    # Save the evaluation metrics in a dataframe and return it
    eval_df = pd.DataFrame({'AUROC': [auroc],
                            'Accuracy': [accuracy],
                            'Precision': [precision],
                            'Recall': [recall],
                            'F1': [f1],
                            'FPR': [fpr],
                            'FNR': [fnr]})

    return eval_df
#%%
fit_logistic_regression(X_train, y_train, X_test, y_test, 0.5, "lr_2")
#lr_1 = pickle.load(open("lr_1", "rb"))

#%%
'''import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())'''


#%%
def plot_classification_performance(df, group_col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    sns.barplot(x=group_col, y='Accuracy', hue='Classifier', data=df, ax=axs[0][0])
    sns.barplot(x=group_col, y='Precision', hue='Classifier', data=df, ax=axs[0][1])
    sns.barplot(x=group_col, y='Recall', hue='Classifier', data=df, ax=axs[1][0])


# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def fit_logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    
    # Fit logistic regression model
    lr = LogisticRegression(random_state=0)
    lr.fit(X_train, y_train)
    
    # Print results
    print("Logistic Regression Results:")
    print("Coefficients:", lr.coef_)
    print("Intercept:", lr.intercept_)
    print("")

    # Save the model
    filename = 'lr_model.sav'
    pickle.dump(lr, open(filename, 'wb'))
    
    return lr

def fit_decision_tree(X_train, y_train, grid_search=False):
    # Set up decision tree classifier
    dt = DecisionTreeClassifier(random_state=0)
    
    # Grid search for hyperparameters
    if grid_search:
        from sklearn.model_selection import GridSearchCV
        params = {'criterion': ['gini', 'entropy'],
                  'max_depth': [3, 5, 7]}
        dt = GridSearchCV(dt, params, cv=5, scoring='roc_auc')
    
    # Fit decision tree model
    dt.fit(X_train, y_train)
    
    # Print results
    print("Decision Tree Results:")
    if grid_search:
        print("Best Parameters:", dt.best_params_)
    print("Feature Importances:", dt.feature_importances_)
    print("")

    # Save the model
    filename = 'dt_model.sav'
    pickle.dump(dt, open(filename, 'wb'))
    
    return dt

def fit_random_forest(X_train, y_train, grid_search=False):
    # Set up random forest classifier
    rf = RandomForestClassifier(random_state=0)
    
    # Grid search for hyperparameters
    if grid_search:
        from sklearn.model_selection import GridSearchCV
        params = {'n_estimators': [50, 100, 200],
                  'criterion': ['gini', 'entropy'],
                  'max_depth': [3, 5, 7]}
        rf = GridSearchCV(rf, params, cv=5, scoring='roc_auc')
    
    # Fit random forest model
    rf.fit(X_train, y_train)
    
    # Print results
    print("Random Forest Results:")
    if grid_search:
        print("Best Parameters:", rf.best_params_)
    print("Feature Importances:", rf.feature_importances_)
    print("")

    # Save the model
    filename = 'rf_model.sav'
    pickle.dump(rf, open(filename, 'wb'))
    
    return rf

def fit_svm(X_train, y_train, grid_search=False):
    # Set up SVM classifier
    svm = SVC(random_state=0, probability=True)
    
    # Grid search for hyperparameters
    if grid_search:
        from sklearn.model_selection import GridSearchCV
        params = {'C': [0.1, 1, 10],
                  'kernel': ['linear', 'rbf', 'sigmoid']}
        svm = GridSearchCV(svm, params, cv=5, scoring='roc_auc')
    
    # Fit SVM model
    svm.fit(X_train, y_train)
    
    # Print results
    print("SVM Results:")
    if grid_search:
        print("Best Parameters:", svm.best_params_)
    print("")

    # Save the model
    filename = 'svm_model.sav'
    pickle.dump(svm, open(filename, 'wb'))
    
