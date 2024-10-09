import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv"
# Alternatively, you can use the other URL provided
# url = "https://storage.googleapis.com/kagglesdsdata/datasets/2491159/4226692/car%20data.csv?..."
car_data = pd.read_csv(url)

# Display the first few rows
print(car_data.head())

# Data exploration
print(car_data.info())
print(car_data.describe())

# Data visualization
sns.pairplot(car_data)
plt.show()

# Check for missing values
print(car_data.isnull().sum())

# Handle categorical variables
car_data = pd.get_dummies(car_data, drop_first=True)

# Split the data into features and target variable
X = car_data.drop('price', axis=1)
y = car_data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'Root Mean Squared Error: {rmse:.2f}')

# Feature importance
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
sns.barplot(x=feature_importances.index, y=feature_importances['importance'])
plt.title('Feature Importances')
plt.xticks(rotation=90)
plt.show()