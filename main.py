from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
# Load the machine learning model
model = joblib.load('house.joblib')
# Define routes and views
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the request
    rooms = request.form.get('rooms')
    bathroom = request.form.get('bathroom')
    landsize = request.form.get('landsize')
    distance = request.form.get('distance')

    # Perform necessary data processing and preprocessing
    # Create a numpy array from the input data
    input_data = np.array([[rooms, bathroom, landsize, distance]])

    # Make predictions using your machine learning model
    predicted_price = model.predict(input_data)
    predicted_pricee = predicted_price.tolist()

    # Return the predicted price as a response
    # return render_template('index.html', prediction=predicted_price)
    # return jsonify(predicted_pricee)
    return jsonify({'predicted_price': predicted_price[0]})


if __name__ == '__main__':
    app.run(debug=True)
