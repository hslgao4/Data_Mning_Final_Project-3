# %% [markdown]
# ## Data Mining Final Project EDA
# ## By: Adewale Maye

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("listings_220611 2.csv")
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

# %%
prices = df['price']

# Plotting histogram
plt.hist(prices, bins=12)  # Adjust number of bins as needed
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Prices')
plt.show()

# %%
# Creating a box plot of 'price' column
plt.boxplot(df['price'])
plt.xlabel('Price')
plt.title('Box Plot of Prices')
plt.show()

# %%
# Creating a scatter plot of 'price' vs 'number_of_reviews' columns
plt.scatter(df['price'], df['review_scores_rating'])
plt.xlabel('Price')
plt.ylabel('Number of Reviews')
plt.title('Scatter Plot of Price vs Number of Reviews')
plt.show()

# %%
# Creating a scatter plot of 'review_scores_rating' vs 'number_of_reviews' columns
plt.scatter(df['review_scores_rating'], df['number_of_reviews'])
plt.xlabel('Review Scores Rating')
plt.ylabel('Number of Reviews')
plt.title('Scatter Plot of Review Scores Rating vs Number of Reviews')
plt.show()

# %%
# Creating a histogram of 'review_scores_rating' column
plt.hist(df['review_scores_rating'], bins=30)
plt.xlabel('Review Scores Rating')
plt.ylabel('Frequency')
plt.title('Histogram of Review Scores Rating')
plt.show()

# %%
# Creating a kernel density plot of 'review_scores_rating' column
plt.hist(df['review_scores_rating'], density=True, alpha=0.6)
df['review_scores_rating'].plot(kind='kde', linewidth=2)
plt.xlabel('Review Scores Rating')
plt.ylabel('Density')
plt.title('Kernel Density Plot of Review Scores Rating')
plt.show()

# %%
# Creating a box plot of 'review_scores_rating' column, grouped by another variable, e.g., 'room_type'
df.boxplot(column='review_scores_rating', by='room_type')
plt.xlabel('Room Type')
plt.ylabel('Review Scores Rating')
plt.title('Box Plot of Review Scores Rating by Room Type')
plt.suptitle('')  # Remove default title
plt.show()

# %%
print(df['review_scores_location'])

# %%
print(df['review_scores_value'])

# %%
# Selecting relevant columns for correlation analysis
cols_for_correlation = ['review_scores_rating', 'accommodates', 'bedrooms', 'beds', 'price', 'min_nights', 'max_nights', 'availability', 'number_of_reviews', 'reviews_per_month']

# Calculating Pearson correlation coefficients
correlation_df = df[cols_for_correlation].corr(method='pearson')

# Plotting correlation heatmap using seaborn
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0)
plt.title('Pearson Correlation Heatmap')
plt.show()

# %%
review_scores_rating_stats = df['review_scores_rating'].describe()
print(review_scores_rating_stats)

# %%
# Create a violinplot of review_scores_rating
sns.violinplot(x='review_scores_rating', data=df)
plt.xlabel('Review Scores Rating')
plt.ylabel('Density')
plt.title('Violinplot of Review Scores Rating')
plt.show()

# %%
# Create a pairplot of review_scores_rating and other numeric variables
sns.pairplot(data=df, vars=['review_scores_rating', 'price', 'bedrooms'])
plt.suptitle('Pairplot of Review Scores Rating and Numeric Variables')
plt.show()


