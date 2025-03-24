import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify, render_template
import os

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target

# Preprocessing
X = df.drop(columns=['PRICE'])
y = df['PRICE']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create Flask App
app = Flask(__name__, template_folder='templates', static_folder='static')

# Ensure templates and static directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Write HTML and CSS for a good-looking interface
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Prediction API</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <h1>House Price Prediction API</h1>
        <p>Enter the feature values below to get a predicted house price.</p>
        <form id="prediction-form">
            <input type="text" id="features" placeholder="Enter 8 comma-separated values" required>
            <button type="submit">Predict</button>
        </form>
        <h2 id="result">Predicted price will appear here.</h2>
        <p id="error" style="color: yellow;"></p>
    </div>
    <script>
        document.getElementById("prediction-form").addEventListener("submit", async function(event) {
            event.preventDefault();
            let input = document.getElementById("features").value;
            let features = input.split(",").map(Number);
            
            if (features.length !== 8 || features.some(isNaN)) {
                document.getElementById("error").innerText = "Please enter exactly 8 valid numbers separated by commas.";
                document.getElementById("result").innerText = "";
                return;
            } else {
                document.getElementById("error").innerText = "";
            }

            let response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ "features": features })
            });
            let data = await response.json();
            document.getElementById("result").innerText = "Predicted Price: $" + data.predicted_price.toFixed(2);
        });
    </script>
</body>
</html>
"""

css_content = """
body {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    font-family: Arial, sans-serif;
    color: white;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
}
.container {
    text-align: center;
    background: rgba(255, 255, 255, 0.2);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
}
input, button {
    padding: 10px;
    margin: 10px;
    border: none;
    border-radius: 5px;
}
input {
    width: 300px;
}
button {
    background: #ff9a9e;
    color: white;
    cursor: pointer;
}
button:hover {
    background: #ff758c;
}
"""

with open('templates/index.html', 'w') as f:
    f.write(html_content)

with open('static/style.css', 'w') as f:
    f.write(css_content)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return an empty response with status code 204 (No Content)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'predicted_price': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
