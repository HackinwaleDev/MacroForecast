import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data into a pandas DataFrame
data = pd.DataFrame({
    'Year': [1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979],
    'Investment': [181.2, 191.2, 205.1, 227.3, 275.6, 297.2, 356.6, 419.8, 469.3, 512.4],
    'Exports': [40.2, 43.2, 48.6, 52.2, 51.8, 61.6, 73.2, 83.3, 91.6, 100.2],
    'Imports': [53.1, 56.5, 63.9, 67.5, 68.1, 72.5, 83.2, 93.6, 106.3, 113.6],
    'Population': [203.2, 206.9, 210.6, 213.9, 217.6, 221.4, 225.2, 228.6, 231.9, 235.5],
    'GDP at 2010 Constant Market Prices': [243.1, 256.8, 264.9, 291.7, 323.8, 334.0, 360.9, 403.1, 443.0, 474.1]
})

# Separate the features (X) and the target variable (y)
X = data.drop(columns=['GDP at 2010 Constant Market Prices'])
y = data['GDP at 2010 Constant Market Prices']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Create a Random Forest regressor
rf = RandomForestRegressor()

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions on the training data
train_predictions = rf.predict(X_train)

# Make predictions on the testing data
test_predictions = rf.predict(X_test)
print(test_predictions)

# Evaluate the model
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
r2 = r2_score(y_test, test_predictions)

# Print the evaluation metrics
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("R^2 Score:", r2)
