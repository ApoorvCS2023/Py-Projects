# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import plotly.graph_objects as go  # For creating advanced visualizations
import plotly.express as px  # For interactive visualizations
import plotly.io as pio  # For handling visualization templates and settings
import opendatasets as od  # For downloading datasets from external sources

# Set the default template for Plotly visualizations to 'plotly_white'
pio.templates.default = 'plotly_white'

# Download the dataset (commented out to prevent re-downloading each time)
# od.download('https://statso.io/wp-content/uploads/2023/07/credit_scoring.csv')

# Load the dataset
data = pd.read_csv('/content/credit_scoring.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Display information about the dataset, such as data types and non-null counts
print(data.info())

# Show descriptive statistics of the dataset to understand the distribution of numerical features
print(data.describe())

# Create a box plot for the distribution of the 'Credit Utilization Ratio'
credit_utilization_fig = px.box(
    data, y='Credit Utilization Ratio', 
    title='Credit Utilization Ratio Distribution'
)
credit_utilization_fig.show()

# Create a histogram for the distribution of the 'Loan Amount'
loan_amount_fig = px.histogram(
    data, x='Loan Amount', nbins=20, 
    title='Loan Amount Distribution'
)
loan_amount_fig.show()

# Select numeric columns for correlation analysis
numeric_df = data[['Credit Utilization Ratio', 'Payment History', 
                   'Number of Credit Accounts', 'Loan Amount', 
                   'Interest Rate', 'Loan Term']]

# Create a heatmap to visualize correlations between numerical features
correlation_fig = px.imshow(numeric_df.corr(), title='Correlation Heatmap')
correlation_fig.show()

# Define mappings for categorical features to numeric values
education_level_mapping = {'High School': 1, 'Bachelor': 2, 
                           'Master': 3, 'PhD': 4}
employment_status_mapping = {'Unemployed': 0, 'Employed': 1, 
                             'Self-Employed': 2}

# Apply the mappings to the categorical columns
data['Education Level'] = data['Education Level'].map(education_level_mapping)
data['Employment Status'] = data['Employment Status'].map(employment_status_mapping)

# Initialize an empty list to store the calculated credit scores
credit_scores = []

# Loop through each row of the dataset to calculate credit scores
for index, row in data.iterrows():
    payment_history = row['Payment History']
    credit_utilization_ratio = row['Credit Utilization Ratio']
    number_of_credit_accounts = row['Number of Credit Accounts']
    education_level = row['Education Level']
    employment_status = row['Employment Status']

    # Calculate the credit score using the FICO formula
    credit_score = (
        (payment_history * 0.35) +
        (credit_utilization_ratio * 0.30) +
        (number_of_credit_accounts * 0.15) +
        (education_level * 0.10) +
        (employment_status * 0.10)
    )
    credit_scores.append(credit_score)

# Add the calculated credit scores as a new column to the DataFrame
data['Credit Score'] = credit_scores

# Display the updated DataFrame with the new 'Credit Score' column
print(data.head())

# Import KMeans for clustering
from sklearn.cluster import KMeans

# Prepare the data for clustering using only the 'Credit Score' column
X = data[['Credit Score']]

# Initialize the KMeans model with 4 clusters
kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)

# Fit the KMeans model to the data
kmeans.fit(X)

# Add the cluster labels to the DataFrame as the 'Segment' column
data['Segment'] = kmeans.labels_

# Convert the 'Segment' column to a categorical data type
data['Segment'] = data['Segment'].astype('category')

# Create a scatter plot to visualize customer segmentation based on credit scores
fig = px.scatter(
    data, x=data.index, y='Credit Score', color='Segment',
    color_discrete_sequence=['green', 'blue', 'yellow', 'red']
)
fig.update_layout(
    xaxis_title='Customer Index',
    yaxis_title='Credit Score',
    title='Customer Segmentation based on Credit Scores'
)
fig.show()

# Map numerical cluster labels to descriptive segment names
data['Segment'] = data['Segment'].map({
    2: 'Very Low', 
    0: 'Low', 
    1: 'Good', 
    3: 'Excellent'
})

# Convert the updated 'Segment' column to a categorical data type
data['Segment'] = data['Segment'].astype('category')

# Create another scatter plot with descriptive segment names
fig = px.scatter(
    data, x=data.index, y='Credit Score', color='Segment',
    color_discrete_sequence=['green', 'blue', 'yellow', 'red']
)
fig.update_layout(
    xaxis_title='Customer Index',
    yaxis_title='Credit Score',
    title='Customer Segmentation based on Credit Scores'
)
fig.show()
