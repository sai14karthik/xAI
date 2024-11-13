import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = None

def load_my_model():
    global model
    try:
        # Load the model only once at the start
        model = load_model('/Users/saikarthik/Desktop/xai /lstm_model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

# Load the model when the app starts
load_my_model()

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Frontend HTML

# Define the upload route
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Get the uploaded CSV file from the form
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file)

        # Print the first few rows for debugging
        print("CSV File Head:\n", df.head())

        # Preprocess the data (assuming data preparation is required)
        # For example, ensure the columns are numeric and prepare them for prediction
        data = df.values.astype(np.float32)  # Convert DataFrame to NumPy array

        # Reshape the data for LSTM input (samples, time steps, features)
        reshaped_data = np.reshape(data, (data.shape[0], 1, data.shape[1]))  # Modify if needed

        # Predict using the model
        prediction = model.predict(reshaped_data)

        # Return prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print(f"Error during file upload or prediction: {str(e)}")
        return jsonify({'error': 'Error processing the file or fetching prediction', 'message': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
