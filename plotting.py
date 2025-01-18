import pandas as pd
import matplotlib.pyplot as plt

def clean_data(file_path, output_file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Define a mapping dictionary for 'Weather Type', 'Season', 'Location', and 'Cloud Cover'
    weather_type_mapping = {
        'cloudy': 0,
        'rainy': 1,
        'sunny': 2,
        'snowy': 3,
    }

    season_mapping = {
        'Winter': 0,
        'Spring': 1,
        'Summer': 2,
        'Autumn': 3
    }

    location_mapping = {
        'inland': 0,
        'coastal': 1,
        'mountain': 2  
    }

    cloud_cover_mapping = {
        'clear': 0,
        'partly cloudy': 1,
        'overcast': 2  
    }

    # Remove rows with missing values
    data_cleaned = data.dropna()

    # Map 'Cloud Cover' column to numeric values
    data_cleaned['Cloud Cover'] = data_cleaned['Cloud Cover'].map(cloud_cover_mapping)

    # Apply the mappings to the categorical columns
    data_cleaned['Weather Type'] = data_cleaned['Weather Type'].map(weather_type_mapping)
    data_cleaned['Season'] = data_cleaned['Season'].map(season_mapping)
    data_cleaned['Location'] = data_cleaned['Location'].map(location_mapping)

    # Save the cleaned data to a new CSV file
    data_cleaned.to_csv(output_file_path, index=False)

    # Return the cleaned dataset
    return data_cleaned

# Path to the original data file (input CSV)
input_file = 'weather_classification_data.csv'  # Adjust if the file is in a different location

# Path where the cleaned data will be saved (output CSV)
output_file = 'weather_classification_data_cleaned.csv'  # Adjust the output file path

# Call the function to clean the data and save it to a new CSV
cleaned_data = clean_data(input_file, output_file)

# Optionally, print or inspect the cleaned data
print("Cleaned data saved to:", output_file)
print(cleaned_data.head())  # Display the first few rows of the cleaned data

# Plot graphs for pattern visualization
plt.figure(figsize=(15, 12))

# Plot Temperature distribution
plt.subplot(3, 3, 1)
plt.hist(cleaned_data['Temperature'], bins=20, color='skyblue', edgecolor='black')
plt.title('Temperature Distribution')
plt.xlabel('Temperature (°C)')
plt.ylabel('Count')

# Plot Humidity distribution
plt.subplot(3, 3, 2)
plt.hist(cleaned_data['Humidity'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Humidity Distribution')
plt.xlabel('Humidity (%)')
plt.ylabel('Count')

# Plot Wind Speed distribution
plt.subplot(3, 3, 3)
plt.hist(cleaned_data['Wind Speed'], bins=20, color='salmon', edgecolor='black')
plt.title('Wind Speed Distribution')
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Count')

# Plot Precipitation distribution
plt.subplot(3, 3, 4)
plt.hist(cleaned_data['Precipitation (%)'], bins=20, color='orange', edgecolor='black')
plt.title('Precipitation Distribution')
plt.xlabel('Precipitation (%)')
plt.ylabel('Count')

# Plot Cloud Cover distribution
plt.subplot(3, 3, 5)
cloud_cover_counts = cleaned_data['Cloud Cover'].value_counts()
plt.bar(cloud_cover_counts.index, cloud_cover_counts.values, color='lightcoral', edgecolor='black')
plt.title('Cloud Cover Distribution')
plt.xlabel('Cloud Cover')
plt.ylabel('Count')

# Plot Atmospheric Pressure distribution
plt.subplot(3, 3, 6)
plt.hist(cleaned_data['Atmospheric Pressure'], bins=20, color='lightblue', edgecolor='black')
plt.title('Atmospheric Pressure Distribution')
plt.xlabel('Pressure (hPa)')
plt.ylabel('Count')

# Plot UV Index distribution
plt.subplot(3, 3, 7)
plt.hist(cleaned_data['UV Index'], bins=20, color='yellow', edgecolor='black')
plt.title('UV Index Distribution')
plt.xlabel('UV Index')
plt.ylabel('Count')

# Plot Visibility distribution
plt.subplot(3, 3, 8)
plt.hist(cleaned_data['Visibility (km)'], bins=20, color='lightpink', edgecolor='black')
plt.title('Visibility Distribution')
plt.xlabel('Visibility (km)')
plt.ylabel('Count')

# Plot Weather Type distribution
plt.subplot(3, 3, 9)
weather_type_counts = cleaned_data['Weather Type'].value_counts()
plt.bar(weather_type_counts.index, weather_type_counts.values, color='purple', edgecolor='black')
plt.title('Weather Type Distribution')
plt.xlabel('Weather Type')
plt.ylabel('Count')

# Adjust layout
plt.tight_layout()
plt.show()
