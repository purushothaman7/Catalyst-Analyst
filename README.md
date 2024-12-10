# Conversion Rate Prediction Model

This project aims to predict the conversion rate of a chemical process using features such as temperature, pressure, surface area, reaction time, and lignin loading. The prediction model leverages a **Random Forest Regressor** trained on experimental data.

## Table of Contents
- [Overview](#overview)
- [Dataset Credits](#dataset-credits)
- [Features](#features)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [License](#license)

## Overview
The goal of this project is to develop a machine learning model that can accurately predict the conversion rate of a chemical reaction based on experimental parameters. The project involves:
- Cleaning and preprocessing a raw dataset.
- Training and testing a regression model.
- Allowing users to predict conversion rates with custom inputs.

## Dataset Credits
Special thanks to **Chandru** and **Pooja**, who meticulously collected and prepared the dataset used in this project. Their work provided the foundation for building and validating the prediction model.

## Features
- **Inputs**: 
  - Temperature (`°C`)
  - Pressure (`MPa`)
  - Surface Area (`m²/g`)
  - Reaction Time (`hours`)
  - Lignin Loading (`wt%`)
- **Output**: Predicted conversion rate as a percentage.
- **Model**: Random Forest Regressor.

## Dataset
The dataset contains experimental data with the following columns:
- `temperature(process)` - Temperature of the process in °C.
- `pressure(process)` - Pressure of the process in MPa.
- `surface_area(m²/g)` - Surface area of the catalyst.
- `reaction_time(h)` - Reaction time in hours.
- `lignin_loading(wt%)` - Lignin loading in weight percent.
- `conversion_rate` - Conversion rate in percentage (target variable).

### Preprocessing
The dataset undergoes the following preprocessing steps:
1. Dropping irrelevant columns (e.g., `paper`, `catalyst used`).
2. Converting non-numeric values to numeric (e.g., temperature, pressure).
3. Parsing mixed-format target values (e.g., percentages or ranges).
4. Handling missing values.

## Dependencies
Make sure to install the following Python libraries before running the code:
- `numpy`
- `pandas`
- `scikit-learn`

You can install them using:
```bash
pip install numpy pandas scikit-learn
```
```bash
git clone https://github.com/purushothaman7/Catalyst-Analyst
cd conversion-rate-prediction
```
Run the Python script:
```bash
python main.py
```
To make predictions, modify the input values in the script

## License
This project is licensed under the MIT License.
