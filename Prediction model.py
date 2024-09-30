# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# import requests
# import pickle

# # Load your historical data
# file_path = r'C:\Users\Jai kattimani\Downloads\Main Tadoba Instagram (1).xlsx'
# df = pd.read_excel(file_path, sheet_name='kada')

# # Convert date column to datetime format
# df['Date_of_safari'] = pd.to_datetime(df['Date_of_safari'])

# # Extract date features
# df['Month'] = df['Date_of_safari'].dt.month
# df['Day'] = df['Date_of_safari'].dt.day
# df['DayOfWeek'] = df['Date_of_safari'].dt.dayofweek

# # Encode Safari_Time
# label_encoder_safari_time = LabelEncoder()
# df['Safari_Time_encoded'] = label_encoder_safari_time.fit_transform(df['Safari_Time'])

# # Encode National Park
# label_encoder_park = LabelEncoder()
# df['National_Park_encoded'] = label_encoder_park.fit_transform(df['National_Park'])

# # Encode Weather Condition
# label_encoder_weather = LabelEncoder()
# df['Weather_Condition_encoded'] = label_encoder_weather.fit_transform(df['Weather_Condition'])

# # Encode Gate (Target variable)
# label_encoder_gate = LabelEncoder()
# df['Gate_encoded'] = label_encoder_gate.fit_transform(df['Gate'])

# # Define features (X) and target (y)
# X = df[['Month', 'Day', 'DayOfWeek', 'Safari_Time_encoded', 'National_Park_encoded', 'Weather_Condition_encoded']]
# y = df['Gate_encoded']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the Decision Tree model
# model = DecisionTreeClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.2f}")

# # Weather API function to fetch forecast for a given city and date
# def get_weather_data(city, country, api_key, date):
#     url = f"http://api.openweathermap.org/data/2.5/forecast?q={city},{country}&units=metric&appid={api_key}"
#     response = requests.get(url)
#     data = response.json()

#     # Extract relevant weather data for the date
#     forecast_data = []
#     for entry in data['list']:
#         forecast_date = entry['dt_txt'].split(' ')[0]  # Extract the date part
#         if forecast_date == date:  # Match the specific date
#             temp = entry['main']['temp']
#             weather_condition = entry['weather'][0]['main']
#             forecast_data.append({
#                 'date': entry['dt_txt'],
#                 'temp': temp,
#                 'condition': weather_condition
#             })
    
#     return forecast_data

# # Function to predict the best gates based on your input column names
# def predict_best_gates(date, national_park_name, safari_time, weather_condition, model):
#     # Extract date features from Date_of_safari
#     month = date.month
#     day = date.day
#     dayofweek = date.dayofweek

#     # Encode the categorical features as per the provided data
#     encoded_safari_time = label_encoder_safari_time.transform([safari_time])[0]  # Encoding Safari_Time
#     encoded_park = label_encoder_park.transform([national_park_name])[0]  # Encoding National_Park
    
#     # Handle unseen weather conditions
#     if weather_condition in label_encoder_weather.classes_:
#         encoded_weather = label_encoder_weather.transform([weather_condition])[0]  # Encoding Weather_Condition
#     else:
#         print(f"Warning: Weather condition '{weather_condition}' not seen in training data. Using fallback label.")
#         encoded_weather = label_encoder_weather.transform(['Cloudy'])[0]  # Fallback condition (use any common label from the training set)

#     # Prepare the input data to pass to the model
#     input_data = [[month, day, dayofweek, encoded_safari_time, encoded_park, encoded_weather]]

#     # Predict probabilities for each gate (Gate column)
#     predicted_probs = model.predict_proba(input_data)[0]

#     # Get the top 3 gates with the highest probabilities
#     top_3_indices = np.argsort(predicted_probs)[-3:][::-1]
#     top_3_gates = label_encoder_gate.inverse_transform(top_3_indices)

#     return top_3_gates

# # Example usage of the prediction function
# date_to_predict = pd.to_datetime('2024-11-30')
# park_to_predict = 'Tadoba Andhari Tiger Reserve'
# safari_time_to_predict = 'Evening'
# weather_condition_to_predict = 'Sunshine'

# best_gates = predict_best_gates(date_to_predict, park_to_predict, safari_time_to_predict, weather_condition_to_predict, model)
# print(f"Best 3 Gates for sightings: {best_gates}")

# # Save the trained model
# with open('gate_prediction_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# # Save the label encoders
# with open('label_encoder_gate.pkl', 'wb') as f:
#     pickle.dump(label_encoder_gate, f)

# with open('label_encoder_park.pkl', 'wb') as f:
#     pickle.dump(label_encoder_park, f)

# with open('label_encoder_weather.pkl', 'wb') as f:
#     pickle.dump(label_encoder_weather, f)

# with open('label_encoder_safari_time.pkl', 'wb') as f:
#     pickle.dump(label_encoder_safari_time, f)


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load your historical data
file_path = r'C:\Users\Jai kattimani\Downloads\Main Tadoba Instagram (1).xlsx'
df = pd.read_excel(file_path, sheet_name='kada')

# Convert date column to datetime format
df['Date_of_safari'] = pd.to_datetime(df['Date_of_safari'])

# Extract date features
df['Month'] = df['Date_of_safari'].dt.month
df['Day'] = df['Date_of_safari'].dt.day
df['DayOfWeek'] = df['Date_of_safari'].dt.dayofweek

# Encode National Park
label_encoder_park = LabelEncoder()
df['National_Park_encoded'] = label_encoder_park.fit_transform(df['National_Park'])

# Encode Gate (Target variable)
label_encoder_gate = LabelEncoder()
df['Gate_encoded'] = label_encoder_gate.fit_transform(df['Gate'])

# Define features (X) and target (y)
X = df[['Month', 'Day', 'DayOfWeek', 'National_Park_encoded']]
y = df['Gate_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to predict the best gates based on your input
def predict_best_gates(date, national_park_name, model):
    # Extract date features from Date_of_safari
    month = date.month
    day = date.day
    dayofweek = date.dayofweek

    # Encode the National Park as per the provided data
    encoded_park = label_encoder_park.transform([national_park_name])[0]

    # Prepare the input data to pass to the model
    input_data = [[month, day, dayofweek, encoded_park]]

    # Predict probabilities for each gate
    predicted_probs = model.predict_proba(input_data)[0]

    # Get the top 3 gates with the highest probabilities
    top_3_indices = np.argsort(predicted_probs)[-3:][::-1]
    top_3_gates = label_encoder_gate.inverse_transform(top_3_indices)

    return top_3_gates


# Example usage of the prediction function
date_to_predict = pd.to_datetime('2024-11-30')
park_to_predict = 'Tadoba Andhari Tiger Reserve'

best_gates = predict_best_gates(date_to_predict, park_to_predict, model)
print(f"Best 3 Gates for sightings: {best_gates}")

# Save the trained model
with open('gate_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the label encoders
with open('label_encoder_gate.pkl', 'wb') as f:
    pickle.dump(label_encoder_gate, f)

with open('label_encoder_park.pkl', 'wb') as f:
    pickle.dump(label_encoder_park, f)
