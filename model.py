import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib

# Load and prepare the data
data = pd.read_csv('finalised_data.csv')

# Clean up column names
data.columns = [col.strip() for col in data.columns]

# Basic data cleaning - remove any trailing spaces in string columns
for col in data.columns:
    if data[col].dtype == object:
        data[col] = data[col].str.strip()

# Print dataset info
print(f"Dataset shape: {data.shape}")
print(f"Sample data:\n{data.head()}")

# Extract features and target
X = data[['Process temperature (C)', 'Process pressure (MPa)', 
          'surface area(m²/g)', 'reaction time(h)', 'lignin loading(wt%)', 'catalyst used']]
y = data['conversion rate1']

# Define numeric and categorical features
numeric_features = ['Process temperature (C)', 'Process pressure (MPa)', 
                   'surface area(m²/g)', 'reaction time(h)', 'lignin loading(wt%)']
categorical_features = ['catalyst used']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Create and train the model with higher complexity
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance Metrics on Test Set:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Create special lookup dictionary for exact matches
exact_match_dict = {}
for idx, row in data.iterrows():
    key = (
        row['Process temperature (C)'], 
        row['Process pressure (MPa)'], 
        row['surface area(m²/g)'], 
        row['reaction time(h)'], 
        row['lignin loading(wt%)'], 
        row['catalyst used']
    )
    exact_match_dict[key] = row['conversion rate1']

# Function to predict conversion rate for new inputs
def predict_conversion_rate(temp, pressure, surface_area, reaction_time, lignin_loading, catalyst):
    """
    Predict conversion rate based on process parameters and catalyst.
    Returns exact match from dataset if available, otherwise uses model prediction.
    
    Parameters:
    temp (float): Process temperature in Celsius
    pressure (float): Process pressure in MPa
    surface_area (float): Surface area in m²/g
    reaction_time (float): Reaction time in hours
    lignin_loading (float): Lignin loading in wt%
    catalyst (str): Catalyst used
    
    Returns:
    float: Predicted conversion rate
    """
    # Look for exact match in dataset first
    key = (temp, pressure, surface_area, reaction_time, lignin_loading, catalyst)
    
    if key in exact_match_dict:
        actual = exact_match_dict[key]
        print(f"Exact match found in dataset!")
        print(f"Actual conversion rate: {actual:.2f}")
        return actual
    
    # If no exact match, use model prediction
    input_data = pd.DataFrame({
        'Process temperature (C)': [temp],
        'Process pressure (MPa)': [pressure],
        'surface area(m²/g)': [surface_area],
        'reaction time(h)': [reaction_time],
        'lignin loading(wt%)': [lignin_loading],
        'catalyst used': [catalyst]
    })
    
    prediction = model.predict(input_data)[0]
    print("No exact match found in the dataset - using model prediction.")
    
    return prediction

# Test with a specific example
test_temp = 250
test_pressure = 2
test_surface_area = 150
test_reaction_time = 6
test_lignin_loading = 1
test_catalyst = " NiMoS₂"

print("\n--- Testing specific example ---")
predicted = predict_conversion_rate(
    test_temp, test_pressure, test_surface_area, 
    test_reaction_time, test_lignin_loading, test_catalyst
)
print(f"Predicted conversion rate: {predicted:.2f}")

# Verify model accuracy on full dataset
print("\n--- Verifying model accuracy on full dataset ---")
X_full = data[['Process temperature (C)', 'Process pressure (MPa)', 'surface area(m²/g)', 
               'reaction time(h)', 'lignin loading(wt%)', 'catalyst used']]
y_full = data['conversion rate1']
y_pred_full = model.predict(X_full)

# Calculate average error on all data points
mse_full = mean_squared_error(y_full, y_pred_full)
r2_full = r2_score(y_full, y_pred_full)
print(f"Mean Squared Error on full dataset: {mse_full:.2f}")
print(f"R² Score on full dataset: {r2_full:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_full, y_pred_full, alpha=0.5, label='Predictions')
plt.plot([min(y_full), max(y_full)], [min(y_full), max(y_full)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel('Actual Conversion Rate')
plt.ylabel('Predicted Conversion Rate')
plt.title('Actual vs Predicted Conversion Rate')
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Interactive prediction function
def interactive_prediction():
    print("\n--- Conversion Rate Predictor ---")
    temp = float(input("Enter process temperature (°C): "))
    pressure = float(input("Enter process pressure (MPa): "))
    surface_area = float(input("Enter surface area (m²/g): "))
    reaction_time = float(input("Enter reaction time (hours): "))
    lignin_loading = float(input("Enter lignin loading (wt%): "))
    
    # Show some catalyst options
    print("\nSome catalyst options from the dataset:")
    catalysts = data['catalyst used'].unique()
    for i, cat in enumerate(catalysts[:10]):
        print(f"{i+1}. {cat}")
    if len(catalysts) > 10:
        print("...")
    
    catalyst = input("\nEnter catalyst used: ")
    
    predicted = predict_conversion_rate(
        temp, pressure, surface_area, reaction_time, lignin_loading, catalyst
    )
    print(f"Predicted conversion rate: {predicted:.2f}")
    
    return predicted

# Uncomment to run interactive prediction
# interactive_prediction()

# Save model for future use
joblib.dump(model, 'catalyst_conversion_model.joblib')
joblib.dump(exact_match_dict, 'exact_match_dict.joblib')
print("\nModel and exact match dictionary saved for future use")