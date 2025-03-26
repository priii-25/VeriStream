import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

state_coords = {
    "Andamanand Nicobar": [11.667, 92.735],
    "Andhra Pradesh": [15.9129, 79.74],
    "Arunachal Pradesh": [28.217, 94.7278],
    "Assam": [26.2006, 92.9376],
    "Bihar": [25.0961, 85.3131],
    "Chhattisgarh": [21.2787, 81.8661],
    "Goa": [15.2993, 74.124],
    "Gujarat": [22.2587, 71.1924],
    "Haryana": [29.0588, 76.0856],
    "Himachal Pradesh": [32.0846, 77.1734],
    "Jharkhand": [23.6102, 85.2799],
    "Karnataka": [15.3173, 75.7139],
    "Kerala": [10.8505, 76.2711],
    "Madhya Pradesh": [23.4735, 77.9479],
    "Maharashtra": [19.6633, 75.3003],
    "Manipur": [24.6637, 93.9063],
    "Meghalaya": [25.467, 91.3662],
    "Mizoram": [23.1645, 92.9376],
    "Nagaland": [26.1584, 94.5624],
    "Odisha": [20.4625, 85.9454],
    "Punjab": [31.1471, 75.3412],
    "Rajasthan": [27.0238, 74.2179],
    "Sikkim": [27.533, 88.5122],
    "TamilNadu": [11.1271, 78.6569],
    "Telangana": [17.1232, 79.2085],
    "Tripura": [23.9408, 91.9882],
    "Uttar Pradesh": [27.1055, 82.0088],
    "Uttarakhand": [30.0668, 79.0193],
    "West Bengal": [22.9868, 87.855],
}

data = pd.read_csv("Misinformation Patterns/indian_misinformation_dataset.csv")

data["latitude"] = data["state"].map(lambda x: state_coords.get(x, [None, None])[0])
data["longitude"] = data["state"].map(lambda x: state_coords.get(x, [None, None])[1])

data = data.dropna(subset=["latitude", "longitude"])

gdf = gpd.read_file("Geospatial Misinformation Mapping/india.json")

merged = gdf.merge(data, left_on="NAME_1", right_on="state")

map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

folium.Choropleth(
    geo_data=gdf,
    name="choropleth",
    data=merged,
    columns=["state", "spread_intensity"],
    key_on="feature.properties.NAME_1",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Spread Intensity",
).add_to(map)

marker_cluster = MarkerCluster().add_to(map)
for _, row in data.iterrows():
    if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=(f"State: {row['state']}<br>"
                   f"Category: {row['category']}<br>"
                   f"Trigger Event: {row['trigger_event']}<br>"
                   f"Fact Check: {row['fact_check_status']}<br>"
                   f"Sentiment Score: {row['sentiment_score']}<br>"
                   f"Verification Confidence: {row['verification_confidence']}"),
            icon=folium.Icon(color="red" if row["fact_check_status"] == "False" else "green"),
        ).add_to(marker_cluster)

folium.LayerControl().add_to(map)
map.save("misinformation_map.html")