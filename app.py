from flask import Flask, render_template, request, jsonify
import pickle
import json

app = Flask(__name__)

# Load the pickled model
with open('house_price_pred.pickle', 'rb') as f:
    model = pickle.load(f)

# Load the columns JSON file
with open('columns.json', 'r') as f:
    columns = json.load(f)

# Define routes
@app.route('/')
def index():
    return render_template('index.html', columns=columns['data_columns'])

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input data
    area = float(request.form['area'])
    bhk = int(request.form['bhk'])
    bathroom = int(request.form['bathroom'])
    per_sqft = float(request.form['per_sqft'])
    selected_columns = request.form.getlist('selected_columns[]')

    # Prepare data for prediction
    input_data = [area, bhk, bathroom, per_sqft] + [1 if col in selected_columns else 0 for col in columns['data_columns'][4:]]

    # Make prediction
    predicted_price = model.predict([input_data])[0]

    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
