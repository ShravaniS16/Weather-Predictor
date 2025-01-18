# Weather-Predictor
A weather prediction system utilizing AI and machine learning techniques. Incorporated expert system approaches alongside machine learning models to analyze historical weather data and generate accurate forecasts, optimizing prediction accuracy and system performance.

# Weather Prediction and Data Analysis

This project leverages machine learning and data visualization techniques to analyze weather data and predict future weather conditions. It processes weather-related features, such as temperature, humidity, wind speed, and more, to predict tomorrow's weather.

## Features

- **Data Cleaning**: Cleans the weather dataset by handling missing values and mapping categorical variables to numeric values.
- **Weather Prediction**: Implements machine learning models to predict the weather based on historical data.
- **Data Visualization**: Utilizes matplotlib to create graphs showing weather patterns and relationships between different variables.
- **User Input for Prediction**: Allows users to input today's weather and predict tomorrow's weather conditions.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow (for ML model)


## Usage

1. **Data Cleaning**: The `clean_data()` function processes the weather data to handle missing values and convert categorical columns to numeric values.
2. **Prediction Model**: After cleaning the data, a machine learning model (using TensorFlow/Keras) is trained to predict the weather based on historical data.
3. **Plotting**: Visualize weather trends with various plots using `matplotlib`.
4. **Weather Prediction Input**: Use the provided interface to input today's weather conditions and predict the weather for the next day.
