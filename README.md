# Airbnb Bookings Analysis - Capstone Project

Since its inception in 2008, Airbnb has transformed the way people experience travel, offering unique and personalized accommodations worldwide. With millions of listings, Airbnb generates vast amounts of data, crucial for various purposes such as security, business decisions, customer behavior analysis, and more. In this project, we delve into the Airbnb NYC 2019 dataset, consisting of around 49,000 observations with 16 columns of categorical and numeric values.

## Table of Contents
1. [Acquiring and Loading Data](#acquiring-and-loading-data)
2. [Data Exploration](#data-exploration)
3. [Variable Identification and Understanding Data](#variable-identification-and-understanding-data)
4. [Handling NaN Values](#handling-nan-values)
5. [Exploring and Visualizing Data](#exploring-and-visualizing-data)
    - [Correlation Matrix](#correlation-matrix)
6. [Single Variable Analysis](#single-variable-analysis)
    - [Top 10 Hosts with the Most Listings](#top-10-hosts-with-the-most-listings)
    - [Neighbourhood Group vs Number of Listings](#neighbourhood-group-vs-number-of-listings)
    - [Top 10 Neighbourhoods in Entire NYC](#top-10-neighbourhoods-in-entire-nyc)
    - [Average Reviews per Month by Top Hosts](#average-reviews-per-month-by-top-hosts)
    - [Average Minimum Nights in Different Room Types](#average-minimum-nights-in-different-room-types)
7. [Bi-variable Analysis](#bi-variable-analysis)
    - [Count of Each Room Type in Neighbourhood Groups](#count-of-each-room-type-in-neighbourhood-groups)
    - [Monthly Reviews Variation with Room Types in Each Neighbourhood Group](#monthly-reviews-variation-with-room-types-in-each-neighbourhood-group)
    - [Room Types and Their Relation with Availability and Neighbourhood Groups](#room-types-and-their-relation-with-availability-and-neighbourhood-groups)
8. ['Price' Feature](#price-feature)
    - [Detecting Outliers Using Boxplot](#detecting-outliers-using-boxplot)
    - [Removing Outliers Using Quantile Approach](#removing-outliers-using-quantile-approach)
    - [Room Types vs Price in Different Neighbourhood Groups](#room-types-vs-price-in-different-neighbourhood-groups)
    - [The Costliest Listings in Entire NYC](#the-costliest-listings-in-entire-nyc)
    - [The Cheapest Listings in Entire NYC](#the-cheapest-listings-in-entire-nyc)
    - [Top Neighbourhoods in Each Neighbourhood Group with Respect to Average Price/Day](#top-neighbourhoods-in-each-neighbourhood-group-with-respect-to-average-price-day)
9. [Conclusion](#conclusion)

## Acquiring and Loading Data
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

drive.mount('/content/drive')
df_airbnb = pd.read_csv('/content/drive/MyDrive/AlmaBetter/Capstone Project/Airbnb NYC 2019 Data.csv')
df_airbnb.head(3)
```

## Data Exploration
```python
df_airbnb.info()
df_airbnb.shape
```

The dataset has 48895 observations with 16 columns. The variables include information about hosts, locations, room types, prices, and more.

## Variable Identification and Understanding Data
```python
df_airbnb.columns
df_airbnb.neighbourhood_group.unique()
df_airbnb['neighbourhood'].nunique()
df_airbnb.room_type.unique()
df_airbnb['host_id'].nunique()
df_airbnb['host_name'].nunique()
df_airbnb['name'].nunique()
df_airbnb.describe()
```

Understanding the dataset involves identifying numerical and categorical variables, exploring unique values, and summarizing numerical variables.

## Handling NaN Values
```python
df_airbnb.drop(['id','last_review'], axis=1, inplace=True)
df_airbnb.fillna({'reviews_per_month':0}, inplace=True)
df_airbnb['name'].fillna('unknown',inplace=True)
df_airbnb['host_name'].fillna('no_name',inplace=True)
```

We address missing values by dropping irrelevant columns and filling NaN values in 'reviews_per_month', 'name', and 'host_name'.

## Exploring and Visualizing Data
```python
# Correlation matrix
plt.figure(figsize=(15,8))
corrmat = df_airbnb.corr()
sns.heatmap(corrmat, vmin=0.0, vmax=1.0, square=True, annot=True, cmap='Spectral');
```

Correlation analysis provides insights into the relationships between different features in the dataset.

## Single Variable Analysis
### Top 10 Hosts with the Most Listings
```python
top_host = df_airbnb.host_id.value_counts

().head(10)
df_top_host = pd.DataFrame(top_host)
df_top_host.reset_index(inplace=True)
df_top_host.rename(columns={'index':'host_id','host_id':'count'},inplace=True)
```

Visualizing the top hosts with the most listings in NYC.
```python
plt.figure(figsize=(15,8))
viz = sns.barplot(x='host_id', y='count', data=df_top_host, palette="PuBu")
viz.set_title('Hosts with the most listings in NYC', size=18)
viz.set_ylabel('Count of listings')
viz.set_xlabel('Host id\'s')
viz.set_xticklabels(viz.get_xticklabels())
plt.show()
```

### Neighbourhood Group vs Number of Listings
```python
plt.figure(figsize=(15,8))
df_airbnb['neighbourhood_group'].value_counts().plot(kind='bar', color='b', rot=0)
plt.xlabel('Neighbourhood Group')
plt.ylabel('Total NYC listings')
plt.title('Count of listings in entire NYC of each neighbourhood group')
plt.show()
```

Visualizing the distribution of listings among different neighbourhood groups in NYC.

### Top 10 Neighbourhoods in Entire NYC
```python
plt.figure(figsize=(15,8))
top_10_neigbours= df_airbnb['neighbourhood'].value_counts()[:10]
top_10_neigbours.plot(kind='bar', color='b', rot=45)
plt.xlabel('Neighbourhood')
plt.ylabel('Counts in entire NYC')
plt.title('Top 10 neighbourhoods in entire NYC on the basis of count of listings')
plt.show()
```

Exploring the top 10 neighbourhoods in NYC based on the count of listings.

### Average Reviews per Month by Top Hosts
```python
# Top 10 most reviewed listings in NYC
top10_reviewed_listings = df_airbnb.nlargest(10,'reviews_per_month')
```

Visualizing the average reviews per month received by the top hosts.
```python
plt.rcParams['figure.figsize'] = (15, 8)
reviews_df = top10_reviewed_listings.groupby('host_name')['reviews_per_month'].mean()
reviews_df = reviews_df.reset_index().sort_values(by='reviews_per_month', ascending=False)
reviews_df.plot(x='host_name', y='reviews_per_month', kind='bar', color='b', rot=0)
plt.ylabel('Reviews counts')
plt.xlabel('Host names')
plt.title('Average Reviews/month received by Top hosts')
plt.show()
```

### Average Minimum Nights in Different Room Types
```python
plt.rcParams['figure.figsize'] = (15, 8)
df_airbnb.groupby('room_type')['minimum_nights'].mean().plot(kind='bar', color='b', rot=0)
plt.title('Min Stays in different room types listed on Airbnb')
plt.ylabel('Min Stays')
plt.xlabel('Room types')
plt.show()
```

## Bi-variable Analysis
### Count of Each Room Type in Neighbourhood Groups
```python
plt.rcParams['figure.figsize'] = (20, 8)
ax = sns.countplot(x='room_type', hue='neighbourhood_group', data=df_airbnb, palette='cubehelix')

total = len(df_airbnb['room_type'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_y() + p.get_height() + 0.02
    ax.annotate(percentage, (x, y), ha='center', bbox=dict(facecolor='white', alpha=.8))

plt.title('Count of each room types in neighbourhood group entire NYC', size=25)
plt.xlabel('Rooms')
plt.xticks(rotation=0)
plt.ylabel('Room Counts')
plt.show()
```

Visualizing the count of each room type in different neighbourhood groups in NYC.

### Monthly Reviews Variation with Room Types in Each Neighbourhood Group
```python
top10_reviewed_listings[['name', 'room_type', 'reviews_per_month', 'neighbourhood_group']]
```

Visualizing the monthly reviews variation with room types in each neighbourhood group.
```python
f, ax = plt.subplots(figsize=(15, 8))
ax = sns.stripplot(x='room_type', y='reviews_per_month', hue='neighbourhood_group', dodge=True, data=df_airbnb, palette='Set2')
ax.set_title('Most Reviewed room types in each Neighbourhood Groups', size=18)
plt.show()
```

### Room Types and Their Relation with Availability and Neighbourhood Groups
```python
# Room Location in neighbourhood groups
plt.figure(figsize=(15, 8))
ax = sns.scatterplot(x='latitude', y='longitude', data=df_airbnb, hue='neighbourhood_group', palette='bright')
ax.set_title('Room Location in neighbourhood groups', size=18)
plt.show()
```

Visualizing the room location in different neighbourhood groups in NYC.
```python
# Distribution of type of rooms across NYC
plt.figure(figsize=(15, 8))
ax = sns.scatterplot(x=df_airbnb['latitude'], y=df_airbnb['longitude'], hue=df_airbnb['room_type'], palette="icefire")
ax.set_title('Distribution of type of rooms across NYC')
plt.show()
```

By the two scatterplots of latitude vs longitude, we can infer there is very less shared room throughout NYC as compared to private and entire home/apt.

95% of the listings on Airbnb are either private room or entire/home apt. Very few guests had opted for shared rooms on Airbnb.

Guests mostly prefer these room types when looking for a rental on Airbnb, as we found out previously in our analysis.

```python
# Listings availability in a year throughout NYC
f, ax = plt.subplots(figsize=(15, 8))
ax = sns.scatterplot(data=df_airbnb, y='longitude', x='latitude', hue="availability_365", palette='RdYlGn', size='availability_365', sizes=(20, 300))
ax.set_title('Listings availability in a year throughout NYC')
plt.show()
```

Bronx and Staten Island have listings that are mostly available throughout the year, possibly due to their lower cost compared to other neighbourhood groups like Manhattan, Brooklyn, and Queens.

## 'Price' Feature
Now let's check the main feature, 'price', and its correlation with other important features.

### Detecting Outliers Using Boxplot
```python
plt.boxplot(df_airbnb['price'])
plt.show()
```

The boxplot shows that the 'price' feature has many outliers. Removing these outliers will provide better results.

### Removing Outliers Using Quantile Approach
```python
min_bound, max_bound = df_airbnb.price.quantile([0.01, 0.999])
df_airbnb[df_airbnb.price < min_bound]
df_airbnb[df_airbnb.price > max_bound]
df_airbnb_new_price = df_airbnb[(df_airbnb.price > min_bound) & (df_airbnb.price < max_bound)]
df_airbnb_new_price.head()
```

### Room Types vs Price in Different Ne

ighbourhood Groups
```python
plt.figure(figsize=(15, 8))
viz_2 = sns.stripplot(x='neighbourhood_group', y='price', data=df_airbnb_new_price, hue='room_type', palette='viridis', dodge=True)
viz_2.set_title('Room Types vs Price in Different Neighbourhood Groups', size=18)
viz_2.set_xlabel('Neighbourhood Groups')
viz_2.set_ylabel('Price')
plt.show()
```

### The Costliest Listings in Entire NYC
```python
df_airbnb_new_price.nlargest(10, 'price')[['name', 'neighbourhood_group', 'price']]
```

### The Cheapest Listings in Entire NYC
```python
df_airbnb_new_price.nsmallest(10, 'price')[['name', 'neighbourhood_group', 'price']]
```

### Top Neighbourhoods in Each Neighbourhood Group with Respect to Average Price/Day
```python
df_airbnb_new_price.groupby(['neighbourhood_group', 'neighbourhood'])[['price']].mean().sort_values(by='price', ascending=False).head(10)
```

## Conclusion
In conclusion, this comprehensive analysis of the Airbnb NYC 2019 dataset provides valuable insights for hosts, guests, and business stakeholders. Understanding the distribution of listings, the preferences of hosts and guests, and the factors influencing prices can contribute to informed decision-making.

This project offers a foundation for further exploration and can be extended with additional data sources and advanced analytical techniques. The findings can be utilized by hosts to optimize their listings, guests to make informed choices, and Airbnb as a platform to enhance user experience and business strategies.
