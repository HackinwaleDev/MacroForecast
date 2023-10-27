import pandas as pd
import mplcursors
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load the Inflation Rate data into a Pandas DataFrame
data = pd.read_csv('_data/MacroEconomicIndicators.csv')

# Separate the features (input variables) and the target variable (Inflation Rate)
X = data.drop(['Year', 'Period', 'Inflation Rate', 'Crude Oil Export'], axis=1)
y = data['Inflation Rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train, X_test)
print('Test Inflation:\n', y_test)

# Create a Random Forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf.fit(X_train, y_train)

# Evaluate the efficiency of the model
trainingScore = rf.score(X_train, y_train)
print(f'Training Score: {trainingScore}')

# Make predictions on the test data
y_pred = rf.predict(X_test)
print("Test Prediction\n", y_pred)


# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Coefficient: {r2}")


# Future prediction:
# You can now use the trained model to predict the Inflation for new input data
new_data = pd.DataFrame([[81.1,	1.27, 460.35,	21827, 527, 17994.28,	18, 33]], 
                        columns=[
                            'Crude Oil Price',	
                            'Production',	
                            # 'Crude Oil Export',		
                            'Exchange Rate',
                            'Government Expenditure',
                            'CPI',	
                            'GDP',		
                            'Interest Rate',
                            'Unemployment Rate'])
predicted_inflation = rf.predict(new_data)
print(f"Predicted Inflation Rate: {predicted_inflation}")

# Visualize the predicted and actual Inflation Rate values
fig, ax = plt.subplots()
ax.plot(range(len(y_test)), y_test, label='Actual Rate')
ax.plot(range(len(y_test)), y_pred, label='Predicted Rate')
ax.set_xlabel('Index')
ax.set_ylabel('Inflation Rate')
ax.set_title('Actual vs Predicted Inflation Rate')
ax.legend()

# Add data value annotations on hover
cursor = mplcursors.cursor(ax, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"Inflation\nRate: {sel.target[1]:.2f}"))

plt.show()
                            