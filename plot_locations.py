import pandas as pd
import folium
from datetime import datetime

# Read the CSV file
df = pd.read_csv('locations_20250324_114600.csv')

# Filter out zero coordinates
df = df[df['lat'] != 0]

# Convert serverTime to datetime for sorting
df['serverTime'] = pd.to_datetime(df['serverTime'])

# Create a map centered on Chennai
chennai_map = folium.Map(
    location=[13.0827, 80.2707],  # Chennai's coordinates
    zoom_start=12,
    tiles='OpenStreetMap'
)

# Get unique device IDs
device_ids = df['deviceId'].unique()

# Create a color map for different devices
import random
import colorsys
def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i/n
        saturation = 0.7
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}')
    return colors

device_colors = {device: color for device, color in zip(device_ids, generate_distinct_colors(len(device_ids)))}

# Plot routes for each device
for device in device_ids:
    # Get data for this device and sort by time
    device_data = df[df['deviceId'] == device].sort_values('serverTime')
    
    # Create route line
    locations = device_data[['lat', 'long']].values.tolist()
    if len(locations) > 1:
        folium.PolyLine(
            locations=locations,
            weight=2,
            color=device_colors[device],
            opacity=0.8,
            dash_array='5',
            popup=f'Route for Device {device}'
        ).add_to(chennai_map)
    
    # Add markers for each location
    for idx, row in device_data.iterrows():
        # Create popup text with device details
        popup_text = f"""
        Device ID: {row['deviceId']}<br>
        Vehicle: {row['vehicleNumber']}<br>
        Speed: {row['speed']} km/h<br>
        Time: {row['serverTime']}
        """
        
        # Add marker with different colors based on speed
        color = 'red' if row['speed'] > 40 else 'green' if row['speed'] > 0 else 'blue'
        
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=5,
            popup=popup_text,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(chennai_map)

# Add a legend
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
    <p><strong>Speed Legend:</strong></p>
    <p><i class="fa fa-circle" style="color: red"></i> > 40 km/h</p>
    <p><i class="fa fa-circle" style="color: green"></i> 0-40 km/h</p>
    <p><i class="fa fa-circle" style="color: blue"></i> Stationary</p>
    <br>
    <p><strong>Device Routes:</strong></p>
'''
for device, color in device_colors.items():
    legend_html += f'<p><i class="fa fa-minus" style="color: {color}"></i> Device {device}</p>'
legend_html += '</div>'

chennai_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map
output_file = f'chennai_bus_routes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
chennai_map.save(output_file)
print(f"Map saved as {output_file}") 