from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import chardet
from charset_normalizer import detect


app = Flask(__name__)

with open("data.csv", "rb") as f:
    result = chardet.detect(f.read())
    print(result)




with open("data.csv", "rb") as f:
    result = detect(f.read())
    print(result)

# Load and preprocess the data
data = pd.read_csv("data.csv")
data = data.drop(columns=['paper ', 'catalyst used'])
data['temperature(process)'] = pd.to_numeric(data['temperature(process)'], errors='coerce')
data['pressure(process)'] = pd.to_numeric(data['pressure(process)'], errors='coerce')

# Clean "conversion rate1" to extract numeric values
def clean_conversion_rate(value):
    if isinstance(value, str):
        value = value.replace('%', '').strip()
        if "to" in value:
            return np.mean([float(v) for v in value.split('to')])
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

data['conversion_rate'] = data['conversion rate1'].apply(clean_conversion_rate)

# Drop the old "conversion rate1" column
data = data.drop(columns=['conversion rate1'])

# Rename columns for consistency
data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('(process)', '').str.lower()

# Drop rows with missing values
data = data.dropna()

# Split the dataset into features (X) and target (y)
X = data.drop(columns=['conversion_rate'])
y = data['conversion_rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Flask route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        temperature = float(request.form['temperature'])
        pressure = float(request.form['pressure'])
        surface_area = float(request.form['surface_area'])
        reaction_time = float(request.form['reaction_time'])
        lignin_loading = float(request.form['lignin_loading'])
        op_catalyst=request.form['op_catalyst']

        # Prepare input as DataFrame
        input_data = pd.DataFrame([[temperature, pressure, surface_area, reaction_time, lignin_loading]],
                                  columns=X_train.columns)

        # Predict the conversion rate
        predicted_conversion_rate = model.predict(input_data)

        # Output prediction
        return render_template('index.html', prediction=predicted_conversion_rate[0], catalyst=op_catalyst)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Get the PORT from environment or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
