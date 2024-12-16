from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

print(sklearn.__version__)

# Loading models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Flask app
app = Flask(__name__)

# Dictionary of regions and crops
regions_and_items = {
    "North": ["Wheat", "Rice", "Barley"],
    "South": ["Sugarcane", "Coconut", "Banana"],
    "East": ["Tea", "Jute", "Rice"],
    "West": ["Cotton", "Groundnut", "Millets"],
}

@app.route('/')
def index():
    return render_template('index.html', regions_and_items=regions_and_items)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area'].strip().capitalize()  # Normalize Area to match dataset format
        Item = request.form['Item'].strip().capitalize()  # Normalize Item to match dataset format

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', prediction=prediction[0][0], regions_and_items=regions_and_items)

if __name__ == "__main__":
    app.run(debug=True)
