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
#dt_2206.head(5)

#%%

#%%
'''filter = dt_2206.host_id == 107434423
temp = dt_2206[filter]
temp'''

#%%
df.head(5)

#%%
'''
#Sample check
filter = df.host_id == 107434423
temp = df[filter]
temp
'''
#%%
#rename PK for join to the initial
dt_2206.rename(columns={'id': 'listing_id'},inplace=True)

#%%
temp = copy.deepcopy(df)

# %%
# 
col = ['listing_id','host_response_time']
temp2 = dt_2206[col]

#%%
#Join with index: 
temp = temp.join(temp2)
#Join with the key

#%%
temp.shape

#%%
'''
Update maindata after creating variables
'''
df = temp
#%%[markdown]
'''
#### `2. Define binary target (review_scores_rating_t2)`
(after 6M, at the moment of Dec 20,2022)
- 0: If review_scores_rating >= 4.5 (High) 
- 1: If review_scores_rating < 4.5 (Low)
- -9: review_scores_rating = NaN (1349, 21.4%),  no review score (number_of_reviews=0 )
      *temporary, convert NaN into -9 (make a decision later)
  
Dataset: make model (0,1) - train 70%,test 30%) 
  
'''

#%%

#Make target on Mar 2023 (review_scores_rating_t2 )
#review_scores_rating_t2(after 6M, at the moment of Dec 20,2022) Will be #joined to the listing of the df (Jun 2022) 

#%%
#Import previous rawdata (containing all cols)
dt_2303 = pd.read_csv("listings_230319.csv")
dt_2303.rename(columns={'id': 'listing_id'},inplace=True) #change pk name
#%%
dt_2303['review_scores_rating_t2'] = dt_2303['review_scores_rating'].apply(lambda x: 0 if x >= 4.5 else 1 if x < 4.5 else -9)

#%%
dt_2303.review_scores_rating_t2.value_counts(dropna=False,normalize=True).sort_index().mul(100).round(1)

#%%
#Left Join review_scores_rating_t2 (review after 6M)
col = ['listing_id','review_scores_rating_t2']
temp2 = dt_2303[col]
df = pd.merge(df, temp2, how='left', on=['listing_id'])
df.tail(5)

#temp: target of current review scores(for finalizing target)
df['review_scores_rating_t'] = df['review_scores_rating'].apply(lambda x: 0 if x >= 4.5 else 1 if x < 4.5 else -9)
#Replace NaN -> -9

pd.crosstab(df['review_scores_rating_t'],df['review_scores_rating_t2'],dropna=False, margins=True, margins_name="Total")

#%%
'''
Finalize target
1)No review on Jun, and still no review after 9 Month (does not have any booking)		
2)No review on Jun, and stop business after 9 Month (No listing)		
> Exclude on modeling, Cound apply to test model (predict their review score for the strategy suggestion) 			
			
3)There was review on Jun but it was deleted that after 9 Month(No review)		
4)There was review on Jun but stop business after 9 Month (No listing)		
> It would be reasonable to assume the reviews on Jun are maintatined			
'''
#Adjust target2 is nan(no listing) or -9(no review) by assumption
temp = copy.deepcopy(df)

#Will change to -9
filter1 = (temp.review_scores_rating_t2.isna()==1) & (temp.review_scores_rating_t ==-9)

#Will change to review_scores_rating_t
filter2= ((temp.review_scores_rating_t2.isna()==1) & (temp.review_scores_rating_t !=-9))|((temp.review_scores_rating_t2==-9) & (temp.review_scores_rating_t !=-9))

#filter3= (temp.review_scores_rating_t2==0)|(temp.review_scores_rating_t2==1)|((temp.review_scores_rating_t==-9)&(temp.review_scores_rating_t2==-9))

#assign
temp.loc[filter1, 'review_scores_rating_t2'] = -9
temp.loc[filter2, 'review_scores_rating_t2'] = temp.loc[filter2, 'review_scores_rating_t']
#temp3 = temp.loc[filter3, :] #no change
temp.shape
#temp.tail(5)

#%%
pd.crosstab(temp['review_scores_rating_t'],temp['review_scores_rating_t2'],dropna=False, margins=True, margins_name="Total")

#%%
'''
Update maindata after creating variables
'''
df = temp
print("End of the target definition. Move on to the next step")

#########################################################

#%%[markdown]
'''
#### `3. Data Cleaning / EDA` 
- Drop unnecessary columns and rename the res<br>
- Convert data type<br>

- Create new columns:<br>
    price_per_room<br>
    Avg_review_comment_score<br>
    years_in_business<br>
    num_amenities<br>

- EDA
'''
# %%
# create "price_per_room"
df['price_per_room'] = df['price'] / df['bedrooms']
# %%
'''
Replace boolean to int (True -> 1, False-> 0)
'''
# transfer data type to bool
def replace_bool(df,var):
    df[var] = df[var].map(lambda x: 1 if x=='t'else 0 if x=='f' else np.nan)

replace_bool(df,'has_availability')
replace_bool(df,'host_has_profile_pic')
replace_bool(df,'host_identity_verified')  
replace_bool(df,'instant_bookable') 
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
       'review_scores_location', 'review_scores_value'
       ], axis=1, inplace=True)
df.shape
#%%
df.info()
# %%
df.iloc[:, :16].head()
#%%
df.iloc[:, 16:32].head()
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
# 2. b
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

    value_counts = df[var].value_counts()
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
# %%
print("\nInformation for 'price_per_room")
print(df["price_per_room"].describe())
print("\n", df["price_per_room"].isnull().sum())

count = df[df['price_per_room'] > 1000]['price_per_room'].count()
print("\nCount for price_per_room over 1000:", count)
df1 = df.drop(df[df['price_per_room'] >= 1000].index)
print('Statistics data after dropping:\n', df1["price_per_room"].describe())

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
plt.xlabel('Price_per_room after droping')
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
plt.hist(df_dropna['review_scores_rating'], density=True, alpha=0.6)
df_dropna['review_scores_rating'].plot(kind='kde', linewidth=2)
plt.xlabel('Review Scores Rating')
plt.ylabel('Density')
plt.title('Kernel Density Plot of Review Scores Rating')
plt.show()
# %%
df_dropna.boxplot(column='review_scores_rating', by='room_type')
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
sns.scatterplot(data = df, x = 'review_scores_rating', y  ='avail_60')

# %% [markdown]
# 9. Correlation check
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
# 10. Chi-square test
from scipy.stats import chi2_contingency

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
df.shape  

#%%
'''
Below is creating additional three features 
   1. Review text mining: Avg_review_comment_score
   2. num_amenities
   3. years_in_business
'''
#%%[markdown]
'''
#### `Join new features 1:  Review text mining(22.6)

- Raw review data(from http://insideairbnb.com/get-the-data): "reviews_220611.csv" 
- number of total reviews: 295978
- number of unique listings: 4958

- Summary of the process
   A. Import raw data, Basic Check (type, null, pk,etc.)
   B. Creating variable: neg, pos, neu
   C. Frequency Check D.Save as pick

- Result: review.pkl (data frame containing 'calculated review score')
   D. Prepare to Join to main dataset: Summarize 'average scores by listing' 
'''

#%%[markdown]
'''
Special treatment for submitting code (30 Apr)
 1. Block code of A~C (since it takes long time to make variables), Active from D.
 2. Modify Result dataframe by dropping 'review comment column' for uploading on Github. 
   (There is size limit when uploading to Github, Rename the revised dataframe as review_sub.pkl)
'''

#%%
'''
A.Import data
'''
#rv = pd.read_csv("reviews_220611.csv")
#rv.shape

# %%
#A. Basic check
#rv.head(5)
#rv.dtypes
#PK check
#rv = rv.sort_values(by='id') 
#rv[rv.id.duplicated()] #no dup
#cleansing null
#rv = rv.dropna()

# %%
#check # of unique listings that have review
#temp = copy.deepcopy(rv)
#temp = temp.sort_values(by='listing_id')  #
#temp = temp.drop_duplicates(subset=['listing_id'])     #delete dup, sort/dup check 
#print(f'number of unique listings: {temp.shape[0]}')

# %%
'''
#B. Creating variable
'''
#import nltk
#nltk.download('vader_lexicon')
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#sid = SentimentIntensityAnalyzer()

# %%
'''
Apply built-in analyzer in the NLTK Python library 
- Assign score of strength(polarity) 
 whether a comment is 'positive'or 'negative' or 'neutral'
'''
'''rv['polarity_value']="Default"
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
rv.head()'''
# %%
'''from langdetect import detect
def detect_lang(sente):
    sente=str(sente)
    try:
        return detect(sente)
    except:
        return "None"

for index,row in rv.iterrows():
    lang=detect_lang(row['comments'])
    rv.at[index,'language'] = lang'''
#     print(lang)
# Check the sort of languages
# print(rv.language.value_counts(dropna=False,normalize=True))
#%%
#taking rows whose language is English
#rv_eng=rv[rv.language=='en']


#%%
'''
C.Frequency Check (10 percentile)
'''
'''polarDF=rv_eng[['pos']]
polarDF=polarDF.groupby(pd.cut(polarDF["pos"], np.arange(0, 1.1, 0.1))).count()
polarDF=polarDF.rename(columns={'pos':'count_of_Comments'})
polarDF=polarDF.reset_index()
polarDF=polarDF.rename(columns={'pos':'range_i'})
for i,r in polarDF.iterrows():
    polarDF.at[i,'RANGE'] = float(str(r['range_i'])[1:4].replace(',',''))
    polarDF.at[i,'Sentiment'] = 'positive'
del polarDF['range_i']
polarDF.head()'''

#%%
'''polarDFneg=rv_eng[['neg']]
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
    
polarDFneg.head()'''

#%%
'''polarDFneut=rv_eng[['neu']]
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
    
polarDFneut.head()'''

#%%
#import seaborn as sns
##%matplotlib inline
#import matplotlib
#import matplotlib.pyplot as plt
#plt.figure(figsize=(10,10))
#sns.scatterplot(x = polarDF['RANGE'], y = polarDF['count_of_Comments'],hue=polarDF['Sentiment']) 

#%%
#print("Save to current dataframe to save as pickle(temp)")
#Sa
#polarDF.to_pickle("./review_sum.pkl")
#rv.to_pickle("./review.pkl")
#rv_eng.to_pickle("./review_eng.pkl")

'''
Save the result
 : Following dataframes are Result of textming(containing 'calculated review score')

rv = pd.read_pickle("./review.pkl")
polarDF = pd.read_pickle("./review_sum.pkl")
rv_eng = pd.read_pickle("./review_eng.pkl") #english only(for example check)
'''

#%%
''''
Examples of negative, postive host.
'''
#%%
#rv_eng.neg.mean()
#filter_1 = rv_eng.neg > 0.4
#filter_2 = rv_eng.pos > 0.4
#col = ['listing_id','comments','neg','pos','neu'] #'pos'

#filter_1 = rv_eng.listing_id == 48113885
#temp = rv_eng[filter_1][col]
#temp.sort_values(by='neg', ascending=False)

#Check sample host in main data (after finishing all process)
#col = ['listing_id','review_scores_rating','number_of_reviews','number_of_reviews_l30d',#'Avg_neg_review_comment_score','Avg_pos_review_comment_score','Avg_neu_review_comment_score']

#filter_1 = df.listing_id == 48113885
#temp = df[filter_1][col]
#temp.T

#%%
'''
This is Modify Result dataframe for submission
 : dropped 'review comment column' for uploading on Github. 
'''
#d = ['comments','reviewer_name']
#review_sub = rv.drop(d, axis=1)
#review_sub.dtypes
#review_sub.to_pickle("./review_sub.pkl")
rv = pd.read_pickle("./review_sub.pkl") #Modify Result dataframe 
rv.language.value_counts().shape #42 languages


#%%
'''
D. Prepare to Join to main dataset: Summarize 'average scores by listing' 
'''

#%%
#print("Load current dataframe and join review features")
#df = pd.read_pickle("./df_clean_v2.pkl") #4/15

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
temp = copy.deepcopy(df)

#%%
print(temp.shape)
temp = pd.merge(temp, pos, how='left', on=['listing_id'])
print(temp.shape)

#%%
print(temp.shape)
temp = pd.merge(temp, neu, how='left', on=['listing_id'])
print(temp.shape)

#%%
print(temp.shape)
temp = pd.merge(temp, neg, how='left', on=['listing_id'])
print(temp.shape)

#%%
print(temp.shape)
temp = pd.merge(temp, neg_dev, how='left', on=['listing_id'])
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
# %% [markdown]
# 8. AVG_NEG_REVIEW_SCORE AND AVG_POS_REVIEW_SCORE
#%%
##BOX PLOT FOR AVG_NEG_REVIEW_SCORE AND AVG_POS_REVIEW_SCORE
sns.boxplot(data=temp[['Avg_neg_review_comment_score', 'Avg_pos_review_comment_score']])
plt.xlabel('Review Comment Score')
plt.ylabel('Score')
plt.title('Box Plot of Avg_neg_review_comment_score vs Avg_pos_review_comment_score')
plt.show()
#%%
##HISTOGRAM FOR AVG_NEG_REVIEW_SCORE AND AVG_POS_REVIEW_SCORE
import matplotlib.pyplot as plt

plt.hist(temp['Avg_neg_review_comment_score'], bins=10, alpha=0.5, label='Avg_neg_review_comment_score')
plt.hist(temp['Avg_pos_review_comment_score'], bins=10, alpha=0.5, label='Avg_pos_review_comment_score')
plt.xlabel('Review Comment Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()


#%%
'''
Update maindata after creating variables
'''
df = temp
df.shape
df.dtypes

#%%
print("End of the 3. [Add features] Review text mining")

###################################################

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
'''
Update maindata after creating variables
'''
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
'''
Update maindata after creating variables
'''
df = temp

#%%
df.dtypes
#%%

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

#%%
print("This is the end of '3. Data Cleaning / EDA'. ")

###############################################
#%%[markdown]
'''
##### `4. Predictive modeling`
Step 1. Final wrangling before modeling<br>
Step 2. Feature selection <br>
(examine Feature Information Gain and Correlation)<br>
Step 3. Train, Test dataset split (Hold-out or Cross validatoin)<br>
Step 4. Fit the models using classficiation algorithms<br>
Step 5. Evaluate models<br>
'''

#%%[markdown]
'''
Step 1. Final wrangling before modeling<br>

 - A. Make Dummy varaibles for category features <br>
 - B. Define modeling data: Target 0 or 1 <br>
     (exclude -9: does not have past review scores)
 - C. Treat NaN value before fitting model <br>

'''
#%%[markdown]
'''
A.cat = [objects] > get_dummy of each categoreis
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

#%%
temp = copy.deepcopy(df)

# %%
temp = make_dummies(temp, cat)

# %%
temp.dtypes

#%%
'''
Update maindata after creating variables
'''
df = temp

#%%
print("Save to current dataframe to save as pickle(temp)")
#df.to_pickle("./df_clean_min.pkl")
#df = pd.read_pickle("./df_clean_min.pkl") #4/15

# %%
df.shape

#%%[markdown]
'''
B.  Define modeling data: Target 0 or 1
'''

#%%

feat = ['listing_id','host_accept.R','profile.pic','identity.verify','bedrooms','beds','reviews_per_month','price_per_room','Avg_neg_review_comment_score','Avg_pos_review_comment_score','Avg_neu_review_comment_score','Std_neg_review_comment_score','Std_pos_review_comment_score','Std_neu_review_comment_score','years_in_business','accommodates','price','min_nights','max_nights','availability','avail_60','number_of_reviews','number_of_reviews_l30d','instant_bookable','calculated_host_listings_count','calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms','num_amenities','room_type_Entire home/apt','room_type_Hotel room','room_type_Private room','room_type_Shared room','host_response_time_a few days or more','host_response_time_within a day','host_response_time_within a few hours','host_response_time_within an hour','review_scores_rating_t2','review_scores_rating_t']

print(f'This is final features that would be used for modeling: {len(feat)} Input variables')
print("- Include target variable and Primary Key")

#%%
df = df[feat]

#%%
#Choose target 0,1 (delete there is no review record)
filter = df.review_scores_rating_t2 != -9 #review_scores_rating_t
df2 = df[filter]
print(df2.shape)

#%%
print(df2.review_scores_rating_t2.value_counts())

#%%[markdown]
'''
C. Treat Null values: Replace it to special value
'''

#%%
print(f'Null value check:{df2.isnull().sum()}')

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
print(f'b. beds=nan -> replace to 1')
df2[df2['beds'].isna()==1].bedrooms.value_counts()
#%%
df2['beds'].value_counts()
#%%
df2['beds'].fillna(1, inplace = True)
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
#df2 = pd.read_pickle("./df_clean_min_fin.pkl") #4/15

#%%
print("*1. Define functions drawing table with vars")

def my_multi_vars(df, var1, var2):
    '''
    p1: freq, p2: ratio(%)
    '''
    p1 = pd.crosstab(df[var1],df[var2], margins=True, margins_name="Total")
    p2 = pd.crosstab(df[var1],df[var2], normalize='index').mul(100).round(1)
    print(f'A.Pivot table {var1} & {var2}(#)\n{p1}\n\nB.Pivot table {var1} & {var2}(%)\n{p2}') 
#%%
my_multi_vars(df2, 'beds', 'review_scores_rating_t2')
#my_multi_vars(df2,'host_accept.R','review_scores_rating_t2')

#%%
print(f'Finished wrangling. Be ready to do modeling.')


#%%[markdown]
'''
#### `4. Predictive modeling
Step 2. Feature selection (examine Feature Information Gain and Correlation) 
'''
#%%
#print("Load final data again")
#df2 = pd.read_pickle("./df_clean_min_fin.pkl") #4/15

print("Calculate Feature importance before Training)")

''''
At the moment, We have 36 candidate input variables. 
They are all numeric. (2 object variables are converted as dummy variables)

#1. Method 1: Information Gain
#2. Method 2: Correlaion between Xs
'''
#%%
#Prepare X,Y for feature selection
X = df2.drop(['review_scores_rating_t','review_scores_rating_t2','listing_id'], axis=1) #All input
y = df2['review_scores_rating_t2']

# %%

''''
#1. Method 1: Information Gain
'''

from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt

def get_information_gain(X, y):
    """
    Compute the information gain of each feature for a binary classification problem.
    
    Args:
    - X: a pandas dataframe containing the features
    - y: a pandas series containing the binary target variable
    
    Returns:
    - A pandas dataframe containing the information gain of each feature
    """
    ig = mutual_info_classif(X, y)
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Information Gain': ig})
    feature_importance.sort_values(by='Information Gain', ascending=False, inplace=True)
    return feature_importance

#%%
ig_df = get_information_gain(X, y)
#print(ig_df)

# plot the feature importance
plt.barh(ig_df['Feature'], ig_df['Information Gain'])
plt.xlabel('Information Gain')
plt.ylabel('Feature')
plt.title('Feature Importance by Information Gain')
plt.show()
#%%
print(ig_df)

#%%

''''
#2. Method 2: Correlaion between Xs
'''

cor = X.corr()
cor
#plt.figure(figsize = (10,6))
#sns.heatmap(cor,annot = True)

#%%

print("Extract features pairs that have correlation over 0.7")

# Extra
high_corr = (cor.abs() > 0.7) & (cor.abs() < 1)
corr_pairs = high_corr.stack().reset_index()
corr_pairs.columns = ['Feature A', 'Feature B', 'Correlation A and B']
corr_pairs = corr_pairs[corr_pairs['Correlation A and B']].sort_values('Correlation A and B', ascending=False)

# Display the resulting dataframe
print(corr_pairs)

#%%

#Initial features: 21 vars, Sorted by feature Information Gain (scending)
ip = ['room_type_Private room','room_type_Hotel room','room_type_Entire home/apt','max_nights','host_accept.R','price','instant_bookable','min_nights','room_type_Shared room','price_per_room','calculated_host_listings_count_entire_homes','avail_60','number_of_reviews_l30d','num_amenities','reviews_per_month','calculated_host_listings_count','number_of_reviews','Avg_neu_review_comment_score','years_in_business','Avg_pos_review_comment_score','Avg_neg_review_comment_score']  #,

print(f'By feature selection, chose {len(ip)} variables that have Feature  Information Gain over 0.01')

#%%[markdown]
'''
#### `4. Predictive modeling
Step 3. Train, Test dataset split (Hold-out or Cross validatoin)
Step 4. Fit the models using classficiation algorithms
Step 5. Evaluate models  
'''

#%%
''''
Step 3. Train, Test dataset split (Hold-out or Cross validatoin)
'''

#%%
print("Load final data again")
#df2 = pd.read_pickle("./df_clean_min_fin.pkl") #4/15

print("By feature selection, we initially chose 21 variables.\n")

print("Choose features and put chosen ones below list,")
#Sorted by feature Information Gain (Descending)
print("Below is the features of the final model(after simulation)")

#Initial 21 vars > best model: logistic_regression_3 (19 vars) 
# > Final model: logistic_regression_22 (exclude 'price', 18 vars)
ip = ['room_type_Entire home/apt','max_nights','host_accept.R','instant_bookable','min_nights','room_type_Shared room','price_per_room','calculated_host_listings_count_entire_homes','avail_60','number_of_reviews_l30d','num_amenities','reviews_per_month','calculated_host_listings_count','number_of_reviews','Avg_neu_review_comment_score','years_in_business','Avg_pos_review_comment_score','Avg_neg_review_comment_score']    #,

X = df2.drop(['review_scores_rating_t','review_scores_rating_t2','listing_id'], axis=1) #All input:'',
y = df2['review_scores_rating_t2']
X = X[ip]

print("Split the Train(70%)/Test data(30%))")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f'1.Low review customer(%)_Train: {round(sum(y_train)/X_train.shape[0]*100,2)}')
print(f'2.Low review customer(%)_Test:{round(sum(y_test)/X_test.shape[0]*100,2)}')


#%%
X.shape[1]
#%%[markdown]
'''
#### `4. Predictive modeling
Step 4. Fit the models using classficiation algorithms
   1. Logistic Regression
   2. Decision Tree
Step 5. Evaluate models  
'''

#%%
''''
# 1. Train Model- Logistic Regression
'''

#%%

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
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


#%%

print("This is the function that fit logistic regresseion using statsmodels package")
#model number
i=1  

def fit_logistic_regression(X_train, y_train, X_test, y_test, cutoff):
    global i, eval_df_total
    
    # Fit logistic regression model on training data
    X_train = sm.add_constant(X_train) # add intercept term
    logit_model = sm.Logit(y_train, X_train)
    lr = logit_model.fit()
    #print(lr.summary())
    
    #fit logistic regresseion using sklearn package
    #lr = LogisticRegression(max_iter=10000)
    #lr.fit(X_train, y_train)
      
    # Save the model
    '''
    Save model file with sequence
    '''
    model_name = f"logistic_regression_{i}.joblib"
    pickle.dump(lr, open(model_name, "wb")) #model_num
    
    # Print model coefficients and p-values
    print("Model Coefficients:")
    print(lr.params)
    print("\nModel Intercept:")
    print(lr.params[0])
    #print("\nModel p-values:")
    #print(lr.pvalues_)

    # Evaluate the model on the test set
    X_test = sm.add_constant(X_test) # add intercept term
    y_pred_prob = lr.predict(X_test)
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
    eval_df = pd.DataFrame({'model_name':[model_name],
                            'AUROC': [auroc],
                            'Cut-off':[cutoff],
                            'Accuracy': [accuracy],
                            'Precision': [precision],
                            'Recall': [recall],
                            'F1': [f1],
                            'FPR': [fpr],
                            'FNR': [fnr]})
    
    # Save the result
    eval_df_total = pd.concat([eval_df_total, eval_df], axis=0).reset_index(drop=True)
    i += 1 # Increase the seq of trial
    return eval_df_total


#%%
# Save the evaluation metrics in a dataframe and return it
eval_df_total = pd.DataFrame({'model_name':[],'AUROC': [],
                        'Cut-off':[],
                        'Accuracy': [],
                        'Precision': [],
                        'Recall': [],
                        'F1': [],
                        'FPR': [],
                        'FNR': []})


#%%
print(f'Fit the single model')

fit_logistic_regression(X_train, y_train, X_test, y_test, 0.5) #0.836710


#%%
'''
(Block: Just show the process)
Simulation of finding optimal logistic regression model. 
 1. Use all features that are initially chosen by feature selection, 21 features.
 2. Re-fit while removing variables one by one ()
'''

#%%
'''X = df2.drop(['review_scores_rating_t','review_scores_rating_t2','listing_id'], axis=1) 
X = X[ip]
for feature in ip:
    y = df2['review_scores_rating_t2']
    X = X.drop(feature, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    fit_logistic_regression(X_train, y_train, X_test, y_test, 0.5)

eval_df_total'''

#%%
'''
#Fit the model that use all 36 variables (for reference)

X = df2.drop(['review_scores_rating_t','review_scores_rating_t2','listing_id'], axis=1) #All 
y = df2['review_scores_rating_t2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
fit_logistic_regression(X_train, y_train, X_test, y_test, 0.5)'''

#%%
print("Wrap-up the logistic regression result)")
eval_df_total
#eval_df_total.to_csv("./Logistic_model_performance.csv")

#%%
print("Interpret the final model result)")
import math
print("0.Load the saved best model:")
test = pickle.load(open("logistic_regression_1.joblib", "rb")) #models/
# Below is Actual model [after simulation: picked #22 model]
# test = pickle.load(open("logistic_regression_22.joblib", "rb"))
print("1.Model Summary Results:")
print(test.summary())
#%%
print("\n2.Model Coefficients:")
print(test.params)
#%%
print("\n3.Exponentiated Coefficients:")
print(pd.Series(np.exp(test.params.values), index=test.params.index))  #get dictionary.value and convert into Series

#%%
print("\n4.AUROC recheck (Trainset):")
X_train = sm.add_constant(X_train) 
y_pred_prob = test.predict(X_train)
#y_pred = (y_pred_prob >= cutoff).astype(int)
 # Calculate evaluation metrics
auroc = roc_auc_score(y_train, y_pred_prob)
auroc

#%%
print(f'Finished logistic regression. Be ready to do decision tree.')
# %%

#%%
''''
# 2. Train Decision Tree
'''
#print("Load final data again")
#df2 = pd.read_pickle("./df_clean_min_fin.pkl") #4/15

#%%
print("By feature selection, we initially chose 21 variables.")

print("Choose features and put chosen ones below list,")
print("Below is the features of the final model(after simulation)")
#Sorted by feature Information Gain (Descending)
#Initial 21 vars > best model: Decision_tree_12 (10 vars) 
ip = ['avail_60','number_of_reviews_l30d','num_amenities','reviews_per_month','calculated_host_listings_count','number_of_reviews','Avg_neu_review_comment_score','years_in_business','Avg_pos_review_comment_score','Avg_neg_review_comment_score'] 

X = df2.drop(['review_scores_rating_t','review_scores_rating_t2','listing_id'], axis=1) #All input:'',
y = df2['review_scores_rating_t2']
X = X[ip]

print("Split the Train(70%)/Test data(30%))")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f'1.Low review customer(%)_Train: {round(sum(y_train)/X_train.shape[0]*100,2)}')
print(f'2.Low review customer(%)_Test:{round(sum(y_test)/X_test.shape[0]*100,2)}')

#%%
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import GridSearchCV

#%%
print("This is the function that fit Decision Tree using sklearn.tree package")
#model number
i=1  

def fit_decision_tree(X_train, y_train, X_test, y_test, cutoff, grid_search):
    global i, eval_df_total_2
    
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()    

    '''
    Grid search for hyperparameters
     : 10 repetitive searches (tree depth, criterion) per each trial 
    '''    
    if grid_search:
        params = {'criterion': ['gini', 'entropy'],
                  'max_depth': [3, 5, 7, 9, 11]}
        clf = GridSearchCV(clf, params, cv=5, scoring='roc_auc')
    
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    
    print("Decision Tree Results:")
    if grid_search:
        print("Best Parameters:", clf.best_params_)
    print("")
      
    # Save the model
    model_name = f"Decision_tree_{i}.joblib"
    pickle.dump(clf, open(model_name, "wb")) #model_num
    
    # Evaluate the model on the test set
    y_pred_prob = clf.predict(X_test)
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
    eval_df = pd.DataFrame({'model_name':[model_name],
                            'Best Parameters':[clf.best_params_],
                            'AUROC': [auroc],
                            'Cut-off':[cutoff],
                            'Accuracy': [accuracy],
                            'Precision': [precision],
                            'Recall': [recall],
                            'F1': [f1],
                            'FPR': [fpr],
                            'FNR': [fnr]})
    
    # Save the result
    eval_df_total_2 = pd.concat([eval_df_total_2, eval_df], axis=0).reset_index(drop=True)
    i += 1 # Increase the seq of trial
    return eval_df_total_2


#%%
# Save the evaluation metrics in a dataframe and return it
eval_df_total_2 = pd.DataFrame({'model_name':[],'Best Parameters':[],
                                'AUROC': [],
                        'Cut-off':[],
                        'Accuracy': [],
                        'Precision': [],
                        'Recall': [],
                        'F1': [],
                        'FPR': [],
                        'FNR': []})

#%%
print("Fit with 10 vars:final model")
fit_decision_tree(X_train, y_train, X_test, y_test, 0.5,True) 
#%%
eval_df_total_2

#%%
'''
Simulation of finding optimal decision tree model. (Treat Block)
 1. Use all features that are initially chosen by feature selection, 21 features.
 2. Re-fit while removing variables one by one ()
'''

#%%
'''X = df2.drop(['review_scores_rating_t','review_scores_rating_t2','listing_id'], axis=1) 
X = X[ip]
for feature in ip:
    y = df2['review_scores_rating_t2']
    X = X.drop(feature, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    fit_decision_tree(X_train, y_train, X_test, y_test, 0.5,True)'''

#%%
#print("Wrap-up the decision tree result)")
#eval_df_total_2
#eval_df_total_2.to_csv("./Decision_Tree_model_performance.csv")    


#%%
print("Interpret the final model result)")
import math
#Load Best model
test = pickle.load(open("Decision_tree_1.joblib", "rb"))
# Below is Actual model [after simulation: picked #12 model]
#test = pickle.load(open("Decision_tree_12.joblib", "rb"))


#%%
'''
'GridSearchCV' object has no attribute 'tree_'
So, ReFit the best model for visualization
'''
#feature list for Decision_tree_12.joblib
ip = ['avail_60','number_of_reviews_l30d','num_amenities','reviews_per_month','calculated_host_listings_count','number_of_reviews','Avg_neu_review_comment_score','years_in_business','Avg_pos_review_comment_score','Avg_neg_review_comment_score']  #,

#%%
X = df2.drop(['review_scores_rating_t','review_scores_rating_t2','listing_id'], axis=1) #All 
X = X[ip]
y = df2['review_scores_rating_t2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(criterion='gini',max_depth=5)
clf = clf.fit(X_train,y_train)

#%%
print("\nAUROC recheck (Trainset):")
y_pred_prob = clf.predict(X_train)
#y_pred = (y_pred_prob >= cutoff).astype(int)
 # Calculate evaluation metrics
auroc = roc_auc_score(y_train, y_pred_prob)
auroc

#%%
#%%
'''
Visualization of decision tree model
'''
from sklearn.tree import export_text
#%%
print("1. Text format of the tree:")
tree_text = export_text(clf, feature_names=ip)
print(tree_text)

#%%
from sklearn import tree
tree.plot_tree(clf)

#%%
print("2. Organized format of the tree")
'''
#Error can be occured depending on version of python. so block it.

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

''# Export the decision tree as a Graphviz dot file
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=dt_col,  
                           class_names=['Higher score', 'Lower score'],  
                           #'Higher score', 'Lower score'. #class_names=TRUE,
                           filled=True, rounded=True,  
                           special_characters=True)

# Display the decision tree using Graphviz
graph = graphviz.Source(dot_data)
#graph.render(filename='decision_tree.pdf')
#graph.view("decision_tree")'''''

#%%


# %%
