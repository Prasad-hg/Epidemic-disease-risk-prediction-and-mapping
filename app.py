from flask import Flask, render_template, request
import pandas as pd
import joblib
import googlemaps
import folium
from folium.plugins import HeatMap
from flask import send_from_directory

app = Flask(__name__)


kmeans = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler_model.pkl')
malaria_model = joblib.load('models/malaria_model.pkl')
dengue_model = joblib.load('models/dengue_model.pkl')


gmaps = googlemaps.Client(key='AIzaSyBDBEHZReEr8Zyc_MKNucPPSUkjMl6YhBA')

data = pd.read_csv('data/high_low_region.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    location_name = request.form['location']
    time_choice = request.form['time_choice']
    value = int(request.form['value'])

    try:
        location = gmaps.geocode(location_name)
        if not location:
            return render_template('result.html', error="Location not found.")
        latitude, longitude = location[0]['geometry']['location']['lat'], location[0]['geometry']['location']['lng']
    except Exception as e:
        return render_template('result.html', error=f"Geocoder error: {e}")

    location_data = data[data['Location'] == location_name]
    if location_data.empty:
        return render_template('result.html', error="Location data not found.")

    avg_features = location_data[['Temperature', 'Humidity', 'Rainfall', 'Malaria Cases', 
                                   'Dengue Cases', 'Population Density', 'Water Body Nearby', 
                                   'Green Cover', 'Healthcare Facilities']].mean()
    cluster = kmeans.predict(scaler.transform([avg_features]))[0]
    risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_level = risk_labels[cluster]

    if time_choice == "month":
        future = pd.DataFrame({'ds': pd.date_range(start=f"2024-{value:02}-01", periods=30, freq='D')})
    elif time_choice == "season":
        start_month = (value - 1) * 3 + 1
        future = pd.DataFrame({'ds': pd.date_range(start=f"2024-{start_month:02}-01", periods=90, freq='D')})

    malaria_forecast = malaria_model.predict(future)
    dengue_forecast = dengue_model.predict(future)

    avg_malaria_cases = malaria_forecast['yhat'].mean()
    avg_dengue_cases = dengue_forecast['yhat'].mean()
    probability_score = min(1.0, (avg_malaria_cases + avg_dengue_cases) / 100)

  
    predictions = [
        [latitude, longitude, probability_score]  
    ]
    generate_heatmap(predictions)

    return render_template(
        'result.html',
        location=location_name,
        risk_level=risk_level,
        probability_score=f"{probability_score:.2f}",
        factors="High Disease Cases, Environmental Conditions"
    )

@app.route('/heatmap')
def heatmap():
    return send_from_directory('output', 'output.html')

def generate_heatmap(predictions):
   
    mapObj = folium.Map(location=[23.294059708387206, 78.26660156250001], zoom_start=6)

  
    bordersStyle = {
        'color': 'green',
        'weight': 1,
        'fillOpacity': 0.1,
    }
    folium.GeoJson('data/states_india.geojson', name='India', style_function=lambda x: bordersStyle).add_to(mapObj)
    folium.GeoJson('data/srilanka.geojson', name='Sri Lanka', style_function=lambda x: bordersStyle).add_to(mapObj)

  
    heatmap_data = []
    for prediction in predictions:
        lat, lon, risk_score = prediction
        heatmap_data.append([lat, lon, risk_score])

    heatmap_layer = HeatMap(heatmap_data, radius=20, blur=15, min_opacity=0.5)
    heatmap_layer.add_to(mapObj)

  
    circle_layer = folium.FeatureGroup(name="Predictions Circles")
    for prediction in predictions:
        lat, lon, risk_score = prediction
        color = 'red' if risk_score >= 0.7 else 'yellow' if risk_score >= 0.4 else 'blue'

   
        folium.Circle(location=[lat, lon],
                      radius=50000,
                      weight=5,
                      color=color,
                      fill_color=color,
                      fill_opacity=0.6,
                      tooltip=f"Risk Level: {risk_score:.2f}",
                      popup=folium.Popup(f"<h2>Risk Prediction: {risk_score:.2f}</h2><p>Latitude: {lat}, Longitude: {lon}</p>", max_width=500)
                     ).add_to(circle_layer)

    circle_layer.add_to(mapObj)

    folium.LayerControl().add_to(mapObj)

    mapObj.save('output/output.html')

if __name__ == '__main__':
    app.run(debug=True)
