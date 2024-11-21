import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('used_cars.csv')

# Display basic info
print(data.info())
print(data.head())

# Clean 'milage' column
data['milage'] = data['milage'].replace({r'[^\d.]': '', r',': ''}, regex=True).astype(float)

# Clean 'engine' column
def clean_engine(value):
    cleaned_value = re.sub(r'[^\d.]', '', str(value))
    if cleaned_value.count('.') > 1:
        cleaned_value = cleaned_value.split('.')[0]
    try:
        return float(cleaned_value)
    except ValueError:
        return 0 

data['engine'] = data['engine'].apply(clean_engine)

# Clean 'accident' column
def clean_accident(value):
    if isinstance(value, str):
        if 'accident' in value.lower() or 'damage' in value.lower():
            return 1
        else:
            return 0
    return value 

data['accident'] = data['accident'].apply(clean_accident)

# Convert 'Yes/No' to binary in 'clean_title' and 'accident'
def convert_yes_no(value):
    if value == 'Yes':
        return 1
    elif value == 'No':
        return 0
    return value

data['clean_title'] = data['clean_title'].apply(convert_yes_no)
data['accident'] = data['accident'].apply(convert_yes_no)

# Clean 'price' column
data['price'] = data['price'].replace({r'[^\d.]': '', r',': ''}, regex=True).astype(float)

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col'], drop_first=True)

# Define features (X) and target variable (y)
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict car prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Create new data for prediction
new_data = pd.DataFrame({
    'model_year': [2020],
    'milage': [15000],
    'engine': [2.0],
    'accident': [0],
    'clean_title': [1], 
}, index=[0])

new_data = new_data.reindex(columns=X.columns, fill_value=0)

# Predict the car price
predicted_price = model.predict(new_data)
print("Predicted Price:", predicted_price)

# Visualization: Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Prices')
plt.grid(True)
plt.show()
