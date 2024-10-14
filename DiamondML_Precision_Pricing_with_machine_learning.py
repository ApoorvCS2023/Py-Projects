# Import necessary libraries
import plotly.express as px  # For creating interactive visualizations
import plotly.graph_objects as go  # For creating complex visualizations
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import opendatasets as od  # For downloading datasets from online sources

# Download the dataset (commented out to avoid re-downloading every time)
# od.download('https://www.kaggle.com/datasets/shivam2503/diamonds')

# Load the diamonds dataset, setting the first column as the index
data = pd.read_csv('diamonds/diamonds.csv', index_col=0)

# Display the first few rows of the dataset to understand the structure
print(data.head())

# Create a scatter plot with carat vs. price, size by depth, colored by cut, and an OLS trendline
figure = px.scatter(data_frame=data, x='carat', y='price', size='depth', 
                    color='cut', trendline='ols')
figure.show()

# Calculate the size of each diamond as volume (x * y * z) and add it to the DataFrame
data['size'] = data['x'] * data['y'] * data['z']

# Display the DataFrame with the new 'size' column
print(data)

# Create a scatter plot with size vs. price, size by the new 'size' column, colored by cut
figure = px.scatter(data_frame=data, x='size', y='price', size='size', 
                    color='cut', trendline='ols')
figure.show()

# Create a box plot to visualize the distribution of price across different cuts and colors
fig = px.box(data, x='cut', y='price', color='color')
fig.show()

# Create another box plot to visualize the distribution of price across different cuts and clarities
fig = px.box(data, x='cut', y='price', color='clarity')
fig.show()

# Convert non-numeric categorical columns to numeric values to calculate correlation
for column in ['cut', 'color', 'clarity']:  # Loop through relevant columns
    try:
        # Try converting the column to numeric
        data[column] = pd.to_numeric(data[column])
    except ValueError:
        # Handle non-numeric values by creating a mapping from categories to numbers
        mapping = {category: i for i, category in enumerate(data[column].unique())}
        data[column] = data[column].map(mapping)  # Apply the mapping to the column

# Calculate the correlation matrix and print the correlations with the price column
correlation = data.corr()
print(correlation['price'].sort_values(ascending=False))

# Map cut quality categories to numeric values for the model
data["cut"] = data["cut"].map({
    "Ideal": 1,
    "Premium": 2,
    "Good": 3,
    "Very Good": 4,
    "Fair": 5
})

# Split the data into training and testing sets (90% training, 10% testing)
from sklearn.model_selection import train_test_split
x = np.array(data[['carat', 'cut', 'size']])  # Features for the model
y = np.array(data['price'])  # Target variable (price)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

# Train a Random Forest Regressor model on the training data
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(xtrain, ytrain)

# Prompt the user to input values for predicting diamond price
print("Diamond Price Prediction")
a = float(input("Carat Size: "))  # Carat size input
b = int(input("Cut Type (Ideal: 1, Premium: 2, Good: 3, Very Good: 4, Fair: 5): "))  # Cut type input
c = float(input("Size (Volume): "))  # Size input (volume)

# Create a feature array from user input and make a prediction using the trained model
features = np.array([[a, b, c]])
print("The Price of the Diamond is: ", model.predict(features))  # Display the predicted price
