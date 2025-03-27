from flask import Flask, render_template, request
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the model and exact match dictionary
model = joblib.load('catalyst_conversion_model.joblib')
exact_match_dict = joblib.load('exact_match_dict.joblib')

# Function to predict conversion rate
def predict_conversion_rate(temp, pressure, surface_area, reaction_time, lignin_loading, catalyst):
    # Look for exact match in dataset first
    key = (temp, pressure, surface_area, reaction_time, lignin_loading, catalyst)
    
    if key in exact_match_dict:
        return {
            'prediction': exact_match_dict[key],
            'match_type': 'exact',
            'message': 'Exact match found in dataset!'
        }
    
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
    return {
        'prediction': prediction,
        'match_type': 'model',
        'message': 'No exact match found - using model prediction'
    }
# Add this new route to your existing app.py

@app.route('/about')
def about():
    team_members = [
        {
            'name': 'Chandru',
            'department': 'Chemical Engineering',
            'role': 'Research Scientist',
            'image': 'chandru.jpg',  # Make sure to add this image to static/images
            'linkedin': 'https://linkedin.com/in/chandru-profile'
        },
        {
            'name': 'Pooja',
            'department': 'Materials Science',
            'role': 'Data Scientist',
            'image': 'pooja.jpg',  # Make sure to add this image to static/images
            'linkedin': 'https://linkedin.com/in/pooja-profile'
        }
    ]
    
    return render_template('about.html', 
                         team_members=team_members,
                         acknowledgement="This website was developed with the assistance of Purushoth, our machine learning expert.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            temp = float(request.form['temperature'])
            pressure = float(request.form['pressure'])
            surface_area = float(request.form['surface_area'])
            reaction_time = float(request.form['reaction_time'])
            lignin_loading = float(request.form['lignin_loading'])
            catalyst = request.form['catalyst']
            
            result = predict_conversion_rate(
                temp, pressure, surface_area, reaction_time, lignin_loading, catalyst
            )
            
            return render_template('result.html', 
                                 prediction=round(result['prediction'], 2),
                                 message=result['message'],
                                 match_type=result['match_type'],
                                 input_data={
                                     'temperature': temp,
                                     'pressure': pressure,
                                     'surface_area': surface_area,
                                     'reaction_time': reaction_time,
                                     'lignin_loading': lignin_loading,
                                     'catalyst': catalyst
                                 })
        except ValueError:
            error = "Please enter valid numbers for all fields"
            return render_template('index.html', error=error)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)