#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#In this assignment, you'll decide which libraries you need to complete the tasks. You can import them in the cell below. 👇

# Import libraries here
import warnings
from glob import glob
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
import plotly.express as px


# In[ ]:


#Prepare Data
#Import
#Task 2.5.1: (8 points) Write a wrangle function that takes the name of a CSV file as input and returns a DataFrame. The function should do the following steps:

#Subset the data in the CSV file and return only apartments in Mexico City ("Distrito Federal") that cost less than $100,000.
#Remove outliers by trimming the bottom and top 10% of properties in terms of "surface_covered_in_m2".
#Create separate "lat" and "lon" columns.
#Mexico City is divided into 16 boroughs. Create a "borough" feature from the "place_with_parent_names" column.
#Drop columns that are more than 50% null values.
#Drop columns containing low- or high-cardinality categorical values.
#Drop any columns that would constitute leakage for the target "price_aprox_usd".
#Drop any columns that would create issues of multicollinearity.
#Tip: Don't try to satisfy all the criteria in the first version of your wrangle function. Instead, work iteratively. Start with the first criteria, test it out with one of the Mexico CSV files in the data/ directory, and submit it to the grader for feedback. Then add the next criteria.

# Build your `wrangle` function
def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Capital Federal", less than 100,000
    mask_ba = df["place_with_parent_names"].str.contains("Distrito Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 100_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Get place name
    df["borough"] = df["place_with_parent_names"].str.split("|", expand=True)[1]
    df.drop(columns="place_with_parent_names", inplace=True)
    
    #Drop features with high null counts
    df.drop(columns=['floor','expenses'],inplace=True)
    
    #Drop High/Low cardinnality catagorital variables
    df.drop(columns=['operation','property_type','currency','properati_url'],inplace=True)
    
    #Drop leaky feature 
    df.drop(columns=[
        'price',
        'price_aprox_local_currency',
        'price_per_m2',
        'price_usd_per_m2'],
        inplace=True
           )
    
    #Drop col with multicollinearity
    df.drop(columns=['surface_total_in_m2','rooms'],inplace=True)
    return df


# In[ ]:


# Use this cell to test your wrangle function and explore the data
df = wrangle('data/mexico-city-real-estate-1.csv')
print("df shape:", df.shape)
df.head(10)


# In[ ]:


#Task 2.5.2: Use glob to create the list files. It should contain the filenames of all the Mexico City real estate CSVs in the ./data directory, except for mexico-city-test-features.csv.

files=glob("data/mexico-city-real-estate-*.csv")
files


# In[ ]:


#Task 2.5.3: Combine your wrangle function, a list comprehension, and pd.concat to create a DataFrame df. It should contain all the properties from the five CSVs in files.

frames = [wrangle(file)for file in files]
df = pd.concat(frames,ignore_index=True)
print(df.info())
df.head()


# In[ ]:


#Explore
#Task 2.5.4: Create a histogram showing the distribution of apartment prices ("price_aprox_usd") in df. Be sure to label the x-axis "Area [sq meters]", the y-axis "Count", and give it the title "Distribution of Apartment Prices". Use Matplotlib (plt).

#What does the distribution of price look like? Is the data normal, a little skewed, or very skewed?

# Build histogram
plt.hist(df['price_aprox_usd'])
# Label axes
plt.xlabel("Area [sq meters]")
plt.ylabel("Count")
# Add title
plt.title("Distribution of Apartment Prices")
# Don't delete the code below 👇
plt.savefig("images/2-5-4.png", dpi=150)


# In[ ]:


#Task 2.5.5: Create a scatter plot that shows apartment price ("price_aprox_usd") as a function of apartment size ("surface_covered_in_m2"). Be sure to label your axes "Price [USD]" and "Area [sq meters]", respectively. Your plot should have the title "Mexico City: Price vs. Area". Use Matplotlib (plt).

# Build scatter plot
plt.scatter(x=df["surface_covered_in_m2"],y=df["price_aprox_usd"])
# Label axes
plt.xlabel("Price [USD]")
plt.ylabel("Area [sq meters]")
# Add title
plt.title("Mexico City: Price vs. Area")
# Don't delete the code below 👇
plt.savefig("images/2-5-5.png", dpi=150)


# In[ ]:


#Task 2.5.6: (UNGRADED) Create a Mapbox scatter plot that shows the location of the apartments in your dataset and represent their price using color.

#What areas of the city seem to have higher real estate prices?

# Plot Mapbox location and price
fig = px.scatter_mapbox(
    df,  # Our DataFrame
    lat='lat',
    lon='lon',
    width=600,  # Width of map
    height=600,  # Height of map
    color='price_aprox_usd',
    hover_data=["price_aprox_usd"],  # Display price when hovering mouse over house
)
fig.update_layout(mapbox_style="open-street-map")
fig.show()


# In[ ]:


#Split
#Task 2.5.7: Create your feature matrix X_train and target vector y_train. Your target is "price_aprox_usd". Your features should be all the columns that remain in the DataFrame you cleaned above.
# Split data into feature matrix `X_train` and target vector `y_train`.

target = "price_aprox_usd"
features =['surface_covered_in_m2','lat','lon',"borough"]
X_train=(df[features])
y_train=(df[target])


# In[ ]:


#Build Model
#Baseline
#Task 2.5.8: Calculate the baseline mean absolute error for your model.
y_mean = y_train.mean()
y_pred_baseline = [y_mean]*len(y_train)
baseline_mae = mean_absolute_error(y_train,y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)


# In[ ]:


#Iterate¶
#Task 2.5.9: Create a pipeline named model that contains all the transformers necessary for this dataset and one of the predictors you've used during this project. Then fit your model to the training data.
# Build Model
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()
)

# Fit model
model.fit(X_train,y_train)


# In[ ]:


#Evaluate
#Task 2.5.10: Read the CSV file mexico-city-test-features.csv into the DataFrame X_test.

#Tip: Make sure the X_train you used to train your model has the same column order as X_test. Otherwise, it may hurt your model's performance.
X_test = pd.read_csv("data/mexico-city-test-features.csv")
y_pred_test = pd.Series(model.predict(X_test))
print(X_test.info())
X_test.head()


# In[ ]:


#Communicate Results
#Task 2.5.12: Create a Series named feat_imp. The index should contain the names of all the features your model considers when making predictions; the values should be the coefficient values associated with each feature. The Series should be sorted ascending by absolute value.
coefficients = model.named_steps['ridge'].coef_.round(2)
features = model.named_steps['ridge']
feat_imp = pd.Series(coefficients,index=feature_names)
feat_imp


# In[ ]:


#Task 2.3.16: Create a horizontal bar chart that shows the top 10 coefficients for your model, based on their absolute value.
feat_imp.sort_values(key=abs).tail(10).plot(kind='barh')
plt.xlabel('Importance [USD]')
plt.ylabel('Feature')
plt.title('Feature Importance for Apartment Price');

