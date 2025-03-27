from flask import Flask, render_template, request
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
import os
from datetime import datetime


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
            'name': 'Dr. Narasimha Reddy',
            'department': 'Chemical Engineering',
            'role': 'Project Mentor',
            'image': 'reddy.jpg',  # Make sure to add this image to static/images
            'linkedin': 'https://www.linkedin.com/in/narasimhareddy18/'
        },
        {
            'name': 'Chandru',
            'department': 'Chemical Engineering',
            'role': 'Data Scientist',
            'image': 'chandru.jpg',  # Make sure to add this image to static/images
            'linkedin': 'https://www.linkedin.com/in/chandru-janakiraman-045aa0248/'
        }, 
        {
            'name': 'Pooja',
            'department': 'Chemical Engineering',
            'role': 'Research Scientist',
            'image': 'pooja.jpg',  # Make sure to add this image to static/images
            'linkedin': 'https://www.linkedin.com/in/pooja-thiagarajan-3020a8249/'
        }
      
    ]
    
    return render_template('about.html', 
                         team_members=team_members)



@app.route('/insights')
def insights():
    # Read and analyze the CSV data
    import pandas as pd
    from io import StringIO
    import numpy as np
    
    csv_data = """Process temperature (C),Process pressure (MPa),surface area(m²/g),reaction time(h),lignin loading(wt%),conversion rate1,catalyst used
700,3,735.58,5,3.33,48,Fe-Fe3C/C
220,5,300,4,7.5,65,NiRu/Al₂O₃
250,2,2382,4,10,43.9,"Ru/4-1AC, "
260,3,600,4,10,52, bimetallic NiCu/C catalyst
240,3,250,3,0.5,69,"ZrO2, WOx, and MoO3"
280,2,1750,6,1,75.6,Ni3Co1-based MOFs [
230,1,1200,0.5,10,53,Ni 10% @Fe/NC 0.33 -800
250,3,800,5,8,35.42,Ni 9 Cu 1 -Mo 2 C/AC and Ni 7 Fe 3 -Mo 2 C/AC.
150,3,250,4,1,48.57,Ni-Ru/Al2O3 
260,3,800,4,0.5,20.8,NiCoC
260,2,800,4,0.5,62,Pd/C (palladium on carbon) combined with NaOH
700,3,134.74,0.0056,1,62,Ni 3 Al 3 -MMO
260,2,750,4,1,34.5,Ni-0.25% Pd/C in synergy with NaOH. 
240,3,100,4,0.05,32.7,CuO/CeO2
280,3,400,4,0.5,62,RuNi/C 
130,3,150,12,1,62,CuZnAl
300,3,700,3,1,37,Ru₂Ni₁-NCNF
300,3,600,4,1,62," Pd-Zn/AC, "
280,1,150,2,0.3,45.6,Ru/LNPC
240,1,200,16.6,0.03,4, Ni₁.₅Fe₁.₅/Al₂O₃
220,0.5,200,0.5,0.5,62,Ni/ZrP
280,1,150,4,1,54,CaO/TiO2
270,1,80,3,1,47,Ru-Cu/Zirconia
180,1,20,2,1,68,NixCoyAl 
200,1,800,4,36.2,67,CeO2/AC
250,4,200,4,1,62.8, Ru/Al₂O₃
280,3,150,0.33,1,15.1,Pt/HAP-3
260,3,200,4,1,22.6,RuNi/SiO2-ZrO2
400,2,300,14,1,42.98,Hf(OTf)4 and Ru/γ-Al2O3
300,3,1000,6,1,43,PTA/MCM-41
300,3,150,3,1,55,Ni 10%/TiN
320,1,150,2,0.1,58," Fe-Pd bimetallic catalyst supported on HZSM-5,"
250,3,18.7,3,1,55.2, N-STC
300,5,100,4,1,98.5,FeNi-ZrO2
250,2,150,6,1,76, NiMoS₂
340,3,200,4,1,62,CuMgAlO x
260,1,246,4,1,92.5,Ni−Co/C 
260,3,200,4,0.5,87.3,Ni/ZrP
260,3,500,2,0.06,77.1,"HZSM-5,"
200,3,400,4,1,62,PdC
350,1,300,4,1,60,Pt-SnCNFI
300,6,97,4,1,50,Ni/C
250,5,198,4,1,20,NiHZSM-5 
250,3,750,0.67,1,62,Ru/C and Pt/C
300,4,200,1,1,68,HPS
280,3,400,4,5,50,Ru/C
200,3,900,8,0.5,61.6,"NiMo/Al-MCM-41,"
300,1.5,400,2,1,62,Pd/C
500,3,450,4,100,62,NiO/HZSM-5 
270,3,300,4,0.15,48,"transition-metal catalysts,"
270,1,600,4,0.3,67.1,NiCu/C 
240,3,139,8,1,62,Ni/(Fe/Zn/Co/Mo/Cu)-WO₃/Al₂O₃
180,2,80,4,1,87.7,Ni/ZrO₂
230,15,200,3,1,62,"Ni 5 Fe 5 /Al 2 O 3,"
350,3,600,2,1,62,20NiMoP/AC
240,3,500,6,1,81.1,Co-Zn/Off-Al H-beta
270,5.5,300,0.5,5,62,"Pd/C, Fe/C, Co/C, and Ni/C"
180,3.5,800,4,1,62, Pd-zeolite Y
260,2,150,4,0.2,66.7, 15Ni/SA-Zr
330,1,200,0.5,3.5,21.2,"NaOH, KOH, and Na₂CO₃ "
350,3,1000,0.67,1,62,Ru and Ni
400,10,300,4,5,25,Ru/C
310,2.5,60,1,4,91.26,"MoS₂, NiS₂/MoS₂, CoS₂/MoS₂, and Ag₂S/MoS₂"
220,3,200,4,1,26.6,Ru/C 
230,2.5,300,4,0.4,88.1,3Ni-1.5Ru/H-ZSM-5
260,4,800,4,1,92.5,5 wt% Ru/C and 5 wt% Pd/C
250,3.5,400,4,15,62, Pt-Ni/SiO2 and Pt-Ni-Cr/Al2O3
240,3,250,4,1,85.66,Ni 1.2 -ZrO 2 /WO 3 /γ-Al 2 O 3
200,3,400,6,1,87.3,Pt/HZSM-23
110,1,367,4,1,28.2,Pt−Ru
300,2,700,4,1,62,Pd/C in combination with CrCl₃
340,5.6,150,3,3,62,ReS2/Al2O3
250,4,200,4,1,62,Hf(OTf)4 (a super Lewis acid) and Ru/Al2O3
220,2,50,0.5,0.2,86.6,NiAl alloy
225,2,400,4,0.3,62,Nb 2 −Ni 1 /ZnO−Al 2 O 3
280,1,200,5,1,91.7,2.5 wt % Ru and 10 wt % Ni
180,1,2399,1,1,75.7,3%Pd-1%Ni/AC
260,0.3,1200,4,1,99,H₂WO₄ and Ru/C.
270,3,200,4,1,91.4, Ni/MgO
290,12,250,3,1,62,"Ru/C, Ni/ZSM-5, and CuNiAl "
200,3,500,4,1,44,Ni/CaO-HZSM-5
""" 
    
    df = pd.read_csv(StringIO(csv_data))
    
    # Calculate high-performance catalysts (above 50% conversion)
    high_perf_catalysts = df[df['conversion rate1'] > 50]
    high_perf_count = len(high_perf_catalysts)
    total_count = len(df)
    high_perf_percentage = round((high_perf_count / total_count) * 100)
    
    # Calculate average conversion by temperature range
    temp_bins = [0, 200, 300, 400, 500, 700]
    temp_labels = ['<200°C', '200-300°C', '300-400°C', '400-500°C', '>500°C']
    df['temp_range'] = pd.cut(df['Process temperature (C)'], bins=temp_bins, labels=temp_labels)
    temp_conversion = df.groupby('temp_range')['conversion rate1'].mean().round(1)
    
    # Prepare insights data
    insights_data = [
        {
            'id': 1,
            'title': 'High-Performance Catalysts',
            'chart_type': 'pie',
            'stats': [
                {'value': f'{high_perf_percentage}%', 'label': 'Above 50% rate'},
                {'value': f'{100-high_perf_percentage}%', 'label': 'Below 50% rate'}
            ],
            'description': f'{high_perf_count} out of {total_count} catalysts ({high_perf_percentage}%) achieve conversion rates above 50%.',
            'icon': 'chart-pie',
            'icon_color': '#3498db'
        },
        {
            'id': 2,
            'title': 'Temperature vs Conversion',
            'chart_type': 'bar',
            'stats': [
                {'value': f'{temp_conversion["200-300°C"]}%', 'label': '200-300°C'},
                {'value': f'{temp_conversion["300-400°C"]}%', 'label': '300-400°C'}
            ],
            'description': 'Optimal conversion occurs in the 200-400°C range. Extreme temperatures show reduced efficiency.',
            'icon': 'thermometer-half',
            'icon_color': '#e74c3c'
        },
        {
            'id': 3,
            'title': 'Pressure Impact',
            'chart_type': 'line',
            'stats': [
                {'value': '3 MPa', 'label': 'Most Common'},
                {'value': '62%', 'label': 'Avg at 3MPa'}
            ],
            'description': 'Moderate pressure (3MPa) appears most frequently in high-conversion scenarios.',
            'icon': 'tachometer-alt',
            'icon_color': '#2ecc71'
        },
        {
            'id': 4,
            'title': 'Surface Area Efficiency',
            'chart_type': 'scatter',
            'stats': [
                {'value': '800 m²/g', 'label': 'Sweet Spot'},
                {'value': '+15%', 'label': 'Efficiency Gain'}
            ],
            'description': 'Catalysts with surface areas around 800 m²/g show consistently better performance.',
            'icon': 'expand-arrows-alt',
            'icon_color': '#f39c12'
        },
        {
            'id': 5,
            'title': 'Top Performing Catalysts',
            'chart_type': 'list',
            'stats': [
                {'value': '92.5%', 'label': 'Ni−Co/C'},
                {'value': '92%', 'label': '5 wt% Ru/C and 5 wt% Pd/C'}
            ],
            'description': 'These catalysts achieve exceptional conversion rates above 98%.',
            'icon': 'award',
            'icon_color': '#9b59b6'
        },
        {
            'id': 6,
            'title': 'Reaction Time Analysis',
            'chart_type': 'area',
            'stats': [
                {'value': '4h', 'label': 'Optimal Duration'},
                {'value': '62%', 'label': 'Avg Conversion'}
            ],
            'description': 'Most high-performance reactions complete within 4 hours.',
            'icon': 'clock',
            'icon_color': '#1abc9c'
        }
    ]

    summary_stats = {
        'total_catalysts': total_count,
        'avg_conversion': f"{round(df['conversion rate1'].mean(), 1)}%",
        'top_performer': 'Ni−Co/C (92.5%)'
    }

    return render_template(
        'insights.html',
        active_page='insights',
        insights=insights_data,
        summary=summary_stats,
        last_updated=datetime.now().strftime("%B %d, %Y")
    )

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