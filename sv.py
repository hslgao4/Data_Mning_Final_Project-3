#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
import pickle 
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
#%%
<<<<<<< Updated upstream
df = pd.read_pickle("D:/Data_Mning_Final_Project-3/df_clean_min.pkl")
=======
df = pd.read_pickle("D:/Data_Mning_Final_Project-3/df_clean_min_fin.pkl")
#%%
df.info()
#%%
##RENAME COLUMNS
new_column_names = {
    'host_accept.R': 'Host_accept',
    'room_type_Entire home/apt': 'RT_EH',
    'room_type_Shared room':'RT_SR',
    'identity.verify':'identity_verify',
    'profile.pic':'profile_pic',
    # Add more mappings as needed
}

# Rename the columns
df.rename(columns=new_column_names, inplace=True)
>>>>>>> Stashed changes
#%%
df.info()
#%%
null_counts = df.isnull().sum()
print(null_counts)
#%%
len(df)
#%%
sns.boxplot(data = df, x = 'room_type', y  ='review_scores_rating')
plt.title(" room type Vs review score : World 1")
plt.xlabel("room_type")
plt.ylabel("review Score")

#%%
sns.boxplot(data = df, x = 'beds', y  ='price')
plt.title(" room type Vs review score : World 1")
plt.xlabel("beds")
plt.ylabel("price")
<<<<<<< Updated upstream
=======
plt.show()
>>>>>>> Stashed changes
#%%
sns.boxplot(data = df, x = 'accommodates', y  ='price')
plt.title(" room type Vs review score : World 1")
plt.xlabel("accomodates")
plt.ylabel("price")
#%%
<<<<<<< Updated upstream
sns.scatterplot(data=df,x="review_scores_rating", y="price")
=======
sns.scatterplot(data=df,x="review_scores_rating_t2", y="price")
>>>>>>> Stashed changes
plt.show()
#%%
sns.scatterplot(data=df,x="review_scores_rating", y="price")
plt.show()
#%%
sns.scatterplot(data=df,x="beds", y="price")
plt.show()
#%%
sns.scatterplot(data = df, x = 'review_scores_rating', y  ='avail_365')
#%%
sns.scatterplot(data = df, x = 'price', y  ='avail_365')
#%%
sns.boxenplot(data = df, x = 'instant_bookable', y  ='avail_365')

#%%
sns.boxenplot(data = df, x = 'instant_bookable', y  ='price')
#%%
cols_for_correlation = ['review_scores_rating','avail_365', 'accommodates', 'bedrooms', 'beds', 'price','availability', 'number_of_reviews', 'reviews_per_month']

# Calculating Pearson correlation coefficients
correlation_df = df[cols_for_correlation].corr(method='pearson')

# Plotting correlation heatmap using seaborn
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0)
plt.title('Pearson Correlation Heatmap')
plt.show()

#%%
df.info()
#%%
df['room_type'].unique()
#%%
from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()
#%%
# Apply label encoding to 'room_type' column
df['room_type_encoded'] = label_encoder.fit_transform(df['room_type'])
#%%
#ONE HOT ENCODING
df = pd.get_dummies(df, columns=['room_type'], prefix='dm')
#%%
df['dm_Entire home/apt']
#%%
df.shape
#%%
df.isna().sum()
#%%
df = df.dropna()
#%%
df.isnull().sum()
#%%
sns.displot(data = df, x ='avail_365')
plt.show()
#%%
sns.displot(data = df, x ='price')
plt.show()
#%%
<<<<<<< Updated upstream
X = df.drop(['review_scores_rating_t2','listing_id'], axis=1) #All input

ip=['bedrooms','beds','price','min_nights','max_nights','availability','avail_365','number_of_reviews','Avg_neg_review_comment_score','Avg_pos_review_comment_score','Avg_neu_review_comment_score']
#%%
X = X[ip]
y = df['review_scores_rating_t2']
#%%
=======
X = df.drop(['review_scores_rating_t','review_scores_rating_t2','listing_id'], axis=1) #All input
y = df['review_scores_rating_t2']
#%%
#SELECT FEATURES
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
print("Define input value)")
X = df.drop(['review_scores_rating_t2','listing_id'], axis=1) #All input

print("Put chosen input values here)")
#ip = ['beds','price_per_room','num_amenities','Avg_neg_review_comment_score','Avg_pos_review_comment_score','Avg_neu_review_comment_score','price','min_nights','avail_60','number_of_reviews','calculated_host_listings_count','years_in_business']
i=[
'Avg_neg_review_comment_score',
'Avg_pos_review_comment_score', 
'Std_neg_review_comment_score',
'years_in_business',
'Std_pos_review_comment_score',
'number_of_reviews',
'Std_neu_review_comment_score',
'Avg_neu_review_comment_score',
'reviews_per_month',
'num_amenities',
'calculated_host_listings_count',
'min_nights',
'price',
'number_of_reviews_l30d',
'Host_accept',
'max_nights',
'avail_60',
'calculated_host_listings_count_shared_rooms',
'price_per_room',
'calculated_host_listings_count_entire_homes',
'calculated_host_listings_count_private_rooms',
'RT_EH',
'RT_SR',
'bedrooms',
'instant_bookable',
'identity_verify',
'profile_pic',
'beds']
#%%

X = X[i]
y = df['review_scores_rating_t2']
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%
>>>>>>> Stashed changes
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=300)
model_fit = model.fit(X_train, y_train)
#%%

print('model accuracy for test', model_fit.score(X_test, y_test))
print('model accuracy for train', model_fit.score(X_train, y_train))
#%%
<<<<<<< Updated upstream
=======
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
#%%
df.info()
# Load your dataset
#%%
log_regression = smf.logit("review_scores_rating_t2 ~ identity.verify + beds + price_per_room + Avg_neg_review_comment_score + Avg_pos_review_comment_score + Avg_neu_review_comment_score + years_in_business + price + min_nights + avail_60+number_of_reviews + calculated_host_listings_count ", data = df)
log_fit = log_regression.fit()
log_fit.summary()
#%%
#FORMULA
formula = "review_scores_rating_t2 ~ " + " + ".join(i)

# Create the logistic regression model
log_regression = smf.logit(formula, data=df)
#%%
#FIT
log_fit = log_regression.fit()
log_fit.summary()
#%%
##DECISION TREE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#%%
clf = DecisionTreeClassifier()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#%%
##BOX PLOT FOR AVG_NEG_REVIEW_SCORE AND AVG_POS_REVIEW_SCORE
import seaborn as sns

sns.boxplot(data=df[['Avg_neg_review_comment_score', 'Avg_pos_review_comment_score']])
plt.xlabel('Review Comment Score')
plt.ylabel('Score')
plt.title('Box Plot of Avg_neg_review_comment_score vs Avg_pos_review_comment_score')
plt.show()
#%%
##HISTOGRAM FOR AVG_NEG_REVIEW_SCORE AND AVG_POS_REVIEW_SCORE
import matplotlib.pyplot as plt

plt.hist(df['Avg_neg_review_comment_score'], bins=10, alpha=0.5, label='Avg_neg_review_comment_score')
plt.hist(df['Avg_pos_review_comment_score'], bins=10, alpha=0.5, label='Avg_pos_review_comment_score')
plt.xlabel('Review Comment Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

>>>>>>> Stashed changes
