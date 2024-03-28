# First, let's load the data to understand its structure, and to identify any duplicates and null values.
import pandas as pd

# Load the CSV file
file_path = 'CarSharing.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

#################################Part 1###################################

# Step 1: Drop duplicate rows
data_deduplicated = data.drop_duplicates()

# Check the shape of the original vs deduplicated data to understand how many duplicates were removed
original_shape = data.shape
deduplicated_shape = data_deduplicated.shape

# Step 2: Identify columns with null values and their counts
null_values_count = data_deduplicated.isnull().sum()

print(original_shape, deduplicated_shape, null_values_count)

# Analyzing the distributions of the columns with null values to decide on the best imputation method
import matplotlib.pyplot as plt

# Plot histograms for each column with null values
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
columns_with_nulls = ['temp', 'temp_feel', 'humidity', 'windspeed']

for col, ax in zip(columns_with_nulls, axes.flatten()):
    data_deduplicated[col].hist(ax=ax, bins=30, alpha=0.7, color='blue')
    ax.set_title(col)

plt.tight_layout()
plt.show()

# Interpolating null values for 'temp', 'temp_feel', 'humidity', and 'windspeed'
data_interpolated = data_deduplicated.copy()
data_interpolated[['temp', 'temp_feel', 'humidity', 'windspeed']] = data_interpolated[['temp', 'temp_feel', 'humidity', 'windspeed']].interpolate(method='linear')

# Verify if there are any null values left
remaining_nulls_after_interpolation = data_interpolated.isnull().sum()

print(remaining_nulls_after_interpolation)

#################################Part 2#####################################

from scipy.stats import pearsonr, f_oneway

# Identifying numerical and categorical columns
numerical_columns = data_interpolated.select_dtypes(include=['float64', 'int64']).columns.drop(['id', 'demand'])
categorical_columns = data_interpolated.select_dtypes(include=['object']).columns.drop(['timestamp'])

# Initializing dictionaries to store test results
pearson_results = {}
anova_results = {}

# Pearson Correlation for Numerical Columns
for col in numerical_columns:
    corr, p_value = pearsonr(data_interpolated[col], data_interpolated['demand'])
    pearson_results[col] = (corr, p_value)

# ANOVA for Categorical Columns
for col in categorical_columns:
    groups = [data_interpolated['demand'][data_interpolated[col] == category] for category in data_interpolated[col].unique()]
    f_stat, p_value = f_oneway(*groups)
    anova_results[col] = (f_stat, p_value)

print(pearson_results, anova_results)

####################################Part 3###############################
import numpy as np

# Step 1: Convert 'timestamp' to datetime format
data_interpolated['timestamp'] = pd.to_datetime(data_interpolated['timestamp'])
#modification

# Step 2: Set 'timestamp' as the index of the DataFrame
data_interpolated.set_index('timestamp', inplace=True)

# Step 3: Extract Data for 2017 (the dataset is already for 2017 based on previous information, so this step is to ensure consistency)
data_2017 = data_interpolated[data_interpolated.index.year == 2017]
#modification

# Select only numeric columns before resampling
numeric_cols = data_2017.select_dtypes(include=[np.number])  # Assuming you have imported NumPy as np

# Then resample and calculate the mean
data_monthly = numeric_cols.resample('M').mean()

'''
# Step 4: Resample Data Monthly to observe trends
data_monthly = data_2017.resample('M').mean()
'''

# Step 5: Plotting the resampled data to identify seasonal or cyclic patterns

fig, axes = plt.subplots(4, 1, figsize=(12, 16))
variables_to_plot = ['temp', 'humidity', 'windspeed', 'demand']

for var, ax in zip(variables_to_plot, axes):
    data_monthly[var].plot(ax=ax, title=var)
    ax.set_ylabel(var)
    ax.grid(True)

plt.tight_layout()
plt.show()

###################################Part 4################################

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Step 1: Prepare Dataset - Convert demand data into weekly averages
weekly_demand = data_interpolated['demand'].resample('W').mean()

# Step 2: Split Dataset into training and testing sets
split_point = int(len(weekly_demand) * 0.7)
train, test = weekly_demand[0:split_point], weekly_demand[split_point:]

# Display the first few entries of the weekly average demand to verify the transformation
print(weekly_demand.head(), train.shape, test.shape)

# Handling NaN value in weekly_demand by filling it with the mean of the series
weekly_demand_filled = weekly_demand.fillna(weekly_demand.mean())

# Update the training and testing sets with the filled data
train_filled, test_filled = weekly_demand_filled[0:split_point], weekly_demand_filled[split_point:]

# Using Auto ARIMA to find the best ARIMA model parameters would be ideal, but educated guesses
# were used for the ARIMA parameters based on typical practices and seasonality observed previously.
# p=1, d=1, q=1, is a common starting point.

# Step 4: Fit ARIMA Model
model = ARIMA(train_filled, order=(1, 1, 2))
model_fit = model.fit()

# Step 5: Make Predictions
predictions = model_fit.forecast(steps=len(test_filled))
predictions_indexed = pd.Series(predictions, index=test_filled.index)

# Step 6: Evaluate Model Performance
rmse = sqrt(mean_squared_error(test_filled, predictions))

print(rmse, predictions_indexed.head())

###################################Part 5###################################

#Random Forest Regressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import numpy as np

# Step 1: Feature Selection and Preprocessing
# Extracting time-related features from 'timestamp' index
data_interpolated['hour'] = data_interpolated.index.hour
data_interpolated['dayofweek'] = data_interpolated.index.dayofweek

# Selecting relevant features and target variable
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'temp_feel', 'humidity', 'windspeed', 'hour', 'dayofweek']
X = data_interpolated[features]
y = data_interpolated['demand']

# Step 2: Encode Categorical Variables
# Identifying categorical columns for one-hot encoding
categorical_features = ['season', 'holiday', 'workingday', 'weather']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preparing the Random Forest pipeline with preprocessing
rf_pipeline = Pipeline(steps=[('preprocessor', transformer),
                              ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

# Training the Random Forest model
rf_pipeline.fit(X_train, y_train)

# Predicting with the Random Forest model
rf_predictions = rf_pipeline.predict(X_test)

# Evaluating the Random Forest model
rf_mse = mean_squared_error(y_test, rf_predictions)

print(rf_mse)

#Deep Neural Network

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Data Preprocessing for DNN
# Scaling numerical features and applying OneHot encoding within a pipeline for DNN
numerical_features = ['temp', 'temp_feel', 'humidity', 'windspeed', 'hour', 'dayofweek']
preprocessor_for_dnn = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)],
    remainder='passthrough')

# Apply preprocessing
X_train_preprocessed = preprocessor_for_dnn.fit_transform(X_train)
X_test_preprocessed = preprocessor_for_dnn.transform(X_test)

# Designing the DNN Architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1) # Output layer for regression
])

# Compiling the DNN
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Training the DNN
model.fit(X_train_preprocessed, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=0)

# Predicting with the DNN
dnn_predictions = model.predict(X_test_preprocessed)

# Evaluating the DNN model
dnn_mse = mean_squared_error(y_test, dnn_predictions)

print(dnn_mse)

#####################################Part 6#########################################

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Step 1: Categorize Demand Rates
average_demand = data_interpolated['demand'].mean()
data_interpolated['demand_label'] = np.where(data_interpolated['demand'] > average_demand, 1, 2)

# Prepare features and target variable for classification
X = data_interpolated[features]  # Using the same features as before
y_class = data_interpolated['demand_label']

# Split the data
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.3, random_state=42)

# Data Preprocessing: Applying the same preprocessing (OneHot encoding for categorical features)
X_train_preprocessed_class = preprocessor_for_dnn.fit_transform(X_train_class)
X_test_preprocessed_class = preprocessor_for_dnn.transform(X_test_class)

# Initialize classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC()
}

# Train, predict, and evaluate each classifier
accuracy_scores = {}
for name, clf in classifiers.items():
    clf.fit(X_train_preprocessed_class, y_train_class)
    predictions = clf.predict(X_test_preprocessed_class)
    accuracy = accuracy_score(y_test_class, predictions)
    accuracy_scores[name] = accuracy

print(accuracy_scores)

###########################################Part 7#############################################

from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np

# Step 1: Prepare the Data
# Extracting temperature data for 2017
temp_data_2017 = data_interpolated[data_interpolated.index.year == 2017]['temp'].values.reshape(-1, 1)  # Reshape for clustering

# Define k values to be evaluated
k_values = [2, 3, 4, 12]

# Initialize dictionaries to store the results
kmeans_results = {}
agglomerative_results = {}

# Perform clustering for each k value using both methods
for k in k_values:
    # K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_clusters = kmeans.fit_predict(temp_data_2017)
    kmeans_cluster_sizes = [list(kmeans_clusters).count(i) for i in range(k)]
    kmeans_results[k] = kmeans_cluster_sizes
    
    # Agglomerative Hierarchical Clustering
    agglomerative = AgglomerativeClustering(n_clusters=k)
    agglomerative_clusters = agglomerative.fit_predict(temp_data_2017)
    agglomerative_cluster_sizes = [list(agglomerative_clusters).count(i) for i in range(k)]
    agglomerative_results[k] = agglomerative_cluster_sizes

# Evaluate uniformity by calculating the standard deviation of cluster sizes for each k
kmeans_uniformity = {k: np.std(sizes) for k, sizes in kmeans_results.items()}
agglomerative_uniformity = {k: np.std(sizes) for k, sizes in agglomerative_results.items()}

print(kmeans_uniformity, agglomerative_uniformity)

