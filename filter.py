import pandas as pd

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
