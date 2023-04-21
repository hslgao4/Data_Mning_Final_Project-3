# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# %%
df = pd.read_pickle("./df_clean_min.pkl")
df.shape

# %%
def replace_bool(df,var):
    df[var] = df[var].map(lambda x: 1 if x=='t'else 0 if x=='f' else np.nan)
#%%
# replace_bool(df,'has_availability')
# replace_bool(df,'host_has_profile_pic')
# replace_bool(df,'host_identity_verified')  
replace_bool(df, 'instant_bookable')

# %%
df.drop(['calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
       'host_is_superhost'], axis=1, inplace=True)
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
print("\n", df["price_per_room"].isnull().sum())
count = df[df['price_per_room'] > 1000]['price_per_room'].count()
print("\n", count)
df = df.drop(df[df['price_per_room'] >= 1000].index)
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


test_inde (df_dropna, "review_scores_rating_t", "profile.pic")
test_inde (df_dropna, "review_scores_rating_t", "identity.verify")
test_inde (df_dropna, "review_scores_rating_t", "availability")
test_inde (df_dropna, "review_scores_rating_t", "instant_bookable")
# %%
data = df_dropna[["review_scores_rating_t","profile.pic", "identity.verify","availability", "instant_bookable" ]]
data.shape
X_train, X_test, y_train, y_test = train_test_split(data.drop("review_scores_rating_t", axis=1), data["review_scores_rating_t"], test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('Accuracy: {:.4f}'.format(accuracy))
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('F1 score: {:.4f}'.format(f1))
print('Confusion matrix:\n{}'.format(cm))
# %% [markdown]
#The accuracy of the model is 0.8749, which means that 87.49% of the predictions made by the model are correct. 
# However, the precision, recall, and F1 score are all 0, which means that the model is not correctly identifying any positive cases.
#
#This could be due to an imbalanced dataset where there are very few positive cases, 
# making it challenging for the model to correctly identify them. 
# Another possibility is that the model is not properly trained and may require additional feature engineering or hyperparameter tuning.

#%%

df_dropna = df_dropna.rename(columns={ 'profile.pic':'profile_pic','identity.verify': 'identity_verify'})
from statsmodels.formula.api import glm
import statsmodels.api as sm

# %%
model_sur_Logit_1 = glm(formula='review_scores_rating_t ~ C(profile_pic)', data=df_dropna, family=sm.families.Binomial())
model_sur_Logit_Fit_1 = model_sur_Logit_1.fit()
print( model_sur_Logit_Fit_1.summary() )
print(model_sur_Logit_Fit_1.null_deviance)
#%%
model_sur_Logit_2 = glm(formula='review_scores_rating_t ~ C(identity_verify)', data=df_dropna, family=sm.families.Binomial())
model_sur_Logit_Fit_2 = model_sur_Logit_2.fit()
print( model_sur_Logit_Fit_2.summary() )
print(model_sur_Logit_Fit_2.null_deviance)
# %%
model_sur_Logit_3 = glm(formula='review_scores_rating_t ~ C(availability)', data=df_dropna, family=sm.families.Binomial())
model_sur_Logit_Fit_3 = model_sur_Logit_3.fit()
print( model_sur_Logit_Fit_3.summary() )
print(model_sur_Logit_Fit_3.null_deviance)
# %%
model_sur_Logit_4 = glm(formula='review_scores_rating_t ~ C(instant_bookable)', data=df_dropna, family=sm.families.Binomial())
model_sur_Logit_Fit_4 = model_sur_Logit_4.fit()
print( model_sur_Logit_Fit_4.summary() )
print(model_sur_Logit_Fit_4.null_deviance)
# %%
model_sur_Logit_5 = glm(formula='review_scores_rating_t ~ C(availability) + C(identity_verify) + C(instant_bookable)', data=df_dropna, family=sm.families.Binomial())
model_sur_Logit_Fit_5 = model_sur_Logit_5.fit()
print( model_sur_Logit_Fit_5.summary() )
print(model_sur_Logit_Fit_5.null_deviance)
# %%
model_sur_Logit_6 = glm(formula='review_scores_rating_t ~ C(availability) + C(identity_verify)', data=df_dropna, family=sm.families.Binomial())
model_sur_Logit_Fit_6 = model_sur_Logit_6.fit()
print( model_sur_Logit_Fit_6.summary() )
print(model_sur_Logit_Fit_6.null_deviance)
# %%
model_sur_Logit_7 = glm(formula='review_scores_rating_t ~ C(availability) + C(instant_bookable)', data=df_dropna, family=sm.families.Binomial())
model_sur_Logit_Fit_7 = model_sur_Logit_7.fit()
print( model_sur_Logit_Fit_7.summary() )
print(model_sur_Logit_Fit_7.null_deviance)
# %%
model_sur_Logit_8 = glm(formula='review_scores_rating_t ~ C(identity_verify) + C(instant_bookable)', data=df_dropna, family=sm.families.Binomial())
model_sur_Logit_Fit_8= model_sur_Logit_8.fit()
print( model_sur_Logit_Fit_8.summary() )
print(model_sur_Logit_Fit_8.null_deviance)
# %%
