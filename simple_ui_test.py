# simple_ui_test.py
import streamlit as st
import pandas as pd
import folium
from datetime import datetime
from south_asian_bird_hotspot import SouthAsianBirdHotspotPredictor

st.title("Bird Hotspot Simple Test")

# Define a test region (Delhi area)
region_bbox = (28.5, 77.0, 28.7, 77.2)
latitude, longitude = 28.6, 77.1

# Run the algorithm
st.write("Running hotspot algorithm...")
predictor = SouthAsianBirdHotspotPredictor(region_bbox, grid_size=0.02)

# Process for current date
with st.spinner("Processing..."):
    result = predictor.process_for_current_date()

# Display results
if result and 'grid' in result and len(result['grid']) > 0:
    st.success(f"Algorithm produced valid results with {len(result['grid'])} grid points")
    
    # Get hotspots
    if 'hotspots' in result and result['hotspots'] is not None:
        st.write(f"Found {len(result['hotspots'])} hotspots")
        
        # Display map
        st.subheader("Hotspot Map")
        m = folium.Map(location=[latitude, longitude], zoom_start=12)
        
        # Add hotspot markers
        for _, row in result['hotspots'].iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=10,
                popup=f"Score: {row['hotspot_score']:.2f}",
                color='red',
                fill=True
            ).add_to(m)
        
        # Display map
        folium_map = folium.Figure().add_child(m)
        st.components.v1.html(folium_map._repr_html_(), height=500)
        
        # Display hotspot table
        st.subheader("Hotspots")
        st.dataframe(result['hotspots'][['latitude', 'longitude', 'hotspot_score']])
else:
    st.error("Algorithm did not produce valid results")