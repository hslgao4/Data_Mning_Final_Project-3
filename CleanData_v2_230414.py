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
temp = temp.join(temp2)
#Join with the key
#df = pd.merge(temp, temp2, how='id')

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
df = pd.read_pickle("./df_clean_v2.pkl") #4/15

#%%
df.shape

#%%
#%%
rv.head(5)

#%%
cols = ['listing_id','neg','pos','neu']
temp = rv[cols]

#%%
neg = temp.groupby('listing_id')["neg"].mean().round(3).reset_index(name='Avg_neg_review_comment_score')
neg.head(5)

#%%
pos = temp.groupby('listing_id')["pos"].mean().round(3).reset_index(name='Avg_pos_review_comment_score')
pos.head(5)

#%%
neu = temp.groupby('listing_id')["neu"].mean().round(3).reset_index(name='Avg_neu_review_comment_score')
neu.head(5)

#%%
print(df.shape)
temp = pd.merge(df, neg, how='left', on=['listing_id'])
print(temp.shape)

#%%
#%%
print(temp.shape)
temp = pd.merge(temp, pos, how='left', on=['listing_id'])
print(temp.shape)

#%%
print(temp.shape)
temp = pd.merge(temp, neu, how='left', on=['listing_id'])
print(temp.shape)

#%%
temp.tail(5)
#%%
df = temp

#%%
df.dtypes
#%%
print("Save to current dataframe to save as pickle(temp)")

#df.to_pickle("./df_clean_min.pkl")
df = pd.read_pickle("./df_clean_min.pkl") #4/15

###############################################
#%%
print("*(Preparation EDA) Drawing pivot table with 2 categorical vars")

def my_multi_vars(df, var1, var2):
    '''
    p1: freq, p2: ratio(%)
    '''
    p1 = pd.crosstab(df[var1],df[var2], margins=True, margins_name="Total")
    p2 = pd.crosstab(df[var1],df[var2], normalize='index').mul(100).round(1)
    print(f'A.Pivot table {var1} & {var2}(#)\n{p1}\n\nB.Pivot table {var1} & {var2}(%)\n{p2}') 

#%%
my_multi_vars(df, 'review_scores_rating_t', 'number_of_reviews')
