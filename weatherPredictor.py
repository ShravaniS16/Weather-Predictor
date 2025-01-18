import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from filter import clean_data

# File paths
input_file_path = 'weather_classification_data.csv'  # Input file path
output_file_path = 'weather_classification_cleaned.csv'  # Output cleaned file path

# Clean the data
data_cleaned = clean_data(input_file_path, output_file_path)

# Check the number of rows in the cleaned dataset
print(f"Number of rows in cleaned dataset: {len(data_cleaned)}")

# Inspect the first few rows of the cleaned data
print("First few rows of cleaned dataset:")
print(data_cleaned.head())

# If the cleaned data is empty or has missing values, handle it
if data_cleaned.isnull().sum().any():
    print("There are still missing values in the cleaned dataset. Handling missing values...")
    data_cleaned = data_cleaned.fillna(0)  # Fill missing values with 0

# Separate the features (X) and target (y)
X = data_cleaned.drop('Weather Type', axis=1)  # Features
y = data_cleaned['Weather Type']  # Target variable

# Ensure that y doesn't contain any NaN values
y = y.fillna(0)

# Split the data into training and testing sets (adjusted test size to 0.1 if needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes: cloudy, rainy, sunny, snowy
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to get user input for today's weather
def get_user_input():
    print("Please enter today's weather details:")

    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    wind_speed = float(input("Wind Speed (km/h): "))
    precipitation = float(input("Precipitation (%): "))
    cloud_cover = int(input("Cloud Cover (0 - clear, 1 - partly cloudy, 2 - overcast): "))
    atmospheric_pressure = float(input("Atmospheric Pressure (hPa): "))
    uv_index = float(input("UV Index: "))
    season = int(input("Season (0 - Winter, 1 - Spring, 2 - Summer, 3 - Autumn): "))
    visibility = float(input("Visibility (km): "))
    location = int(input("Location (0 - inland, 1 - coastal, 2 - mountain): "))

    # Create a DataFrame for the user input
    user_input = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Wind Speed': [wind_speed],
        'Precipitation (%)': [precipitation],
        'Cloud Cover': [cloud_cover],
        'Atmospheric Pressure': [atmospheric_pressure],
        'UV Index': [uv_index],
        'Season': [season],
        'Visibility (km)': [visibility],
        'Location': [location]
    })
    
    return user_input

# Get user input for today's weather
example_input = get_user_input()

# Predict the weather type for tomorrow based on the user input
predicted_weather_probs = model.predict(example_input)
predicted_weather = tf.argmax(predicted_weather_probs, axis=1).numpy()

# Map the predicted weather type to its corresponding label
weather_types = {0: 'cloudy', 1: 'rainy', 2: 'sunny', 3: 'snowy'}
predicted_weather_type = weather_types[predicted_weather[0]]

print(f"Predicted weather type for tomorrow: {predicted_weather_type}")
