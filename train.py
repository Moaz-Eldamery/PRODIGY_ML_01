import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np



# load the training data using pandas
train_data = pd.read_csv('house-prices-advanced-regression-techniques\\train.csv')
test_data = pd.read_csv('house-prices-advanced-regression-techniques\\test.csv')

# select relevant features and target variable(square footage, number of bedrooms, number of bathrooms)
features = ['Id','GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

# drop rows with missing values in the selected columns
data = train_data[features + [target]].dropna()
print(data.head)

# Split the data into features (X) and target (y)
X_train = data[features]
y_train = data[[target]]
X_test = test_data[features]

# Split the data into training and testing sets

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model


# this is to display the data while developing the code

# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted vs. Actual')
# max_price = max(max(y_test), max(y_pred))
# min_price = min(min(y_test), min(y_pred))
# plt.plot([min_price, max_price], [min_price, max_price], color='red', linestyle='--', label='Perfect Fit')
# plt.title('Actual vs. Predicted House Prices')
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.legend()
# plt.grid(True)
# plt.show()

# Display some predictions alongside actual prices to make sure of the accurecy
results = X_test.copy()
results['Predicted Price'] = y_pred

print(results.head())



results['Id']= test_data.loc[X_test.index, 'Id']

# creating the subbmission file
submission = results[['Id', 'Predicted Price']]

# Save the submission to a CSV file
submission.to_csv('house-prices-advanced-regression-techniques\\submission.csv', index=False)
