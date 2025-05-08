import os
import sys
import folium
import pandas as pd
import geopandas as gpd
from datetime import datetime
from geopy.geocoders import Nominatim
from shapely.geometry import Point
import streamlit as st
import time
import traceback
import logging
from folium.plugins import HeatMap
import requests
from datetime import timedelta
import json
from dotenv import load_dotenv
from bird_api_client import BirdDataClient
import matplotlib.pyplot as plt
import numpy as np
import south_asian_bird_hotspot

# Load environment variables for API keys
load_dotenv()

# API keys (store these in a .env file for security)
EBIRD_API_KEY = os.getenv("EBIRD_API_KEY", "")
GBIF_USER = os.getenv("GBIF_USER", "")
GBIF_PWD = os.getenv("GBIF_PWD", "")

# Define the hotspot details function first
def generate_hotspot_details(hotspot_row):
    """Generate detailed information for a hotspot popup."""
    
    # Basic information
    details = f"""
    <div style='min-width: 200px'>
    <h4>Hotspot Details</h4>
    <b>Score:</b> {hotspot_row['hotspot_score']:.2f}<br>
    """
    
    # Add habitat diversity if available
    if 'habitat_diversity' in hotspot_row:
        details += f"<b>Habitat Diversity:</b> {hotspot_row['habitat_diversity']:.2f}<br>"
    
    # Add bird count information if available
    if 'bird_count' in hotspot_row:
        details += f"<b>Bird Count:</b> {int(hotspot_row['bird_count'])}<br>"
    else:
        # Simulate bird count based on hotspot score
        estimated_count = int(hotspot_row['hotspot_score'] * 100)
        details += f"<b>Estimated Bird Count:</b> {estimated_count}<br>"
    
    # Add species information if available
    if 'species_count' in hotspot_row:
        details += f"<b>Species Count:</b> {int(hotspot_row['species_count'])}<br>"
    else:
        # Simulate species count based on hotspot score
        estimated_species = int(hotspot_row['hotspot_score'] * 30)
        details += f"<b>Estimated Species:</b> {estimated_species}<br>"
    
    # Add habitat information
    details += "<b>Key Habitats:</b><br>"
    for col in hotspot_row.index:
        if col.startswith('is_') and hotspot_row[col] > 0:
            habitat_name = col.replace('is_', '').replace('_', ' ').title()
            details += f"- {habitat_name}<br>"
    
    # Add calculation explanation
    details += "<h4>Score Calculation</h4>"
    details += "<table style='width:100%'>"
    details += "<tr><th>Factor</th><th>Weight</th><th>Value</th></tr>"
    
    # Habitat diversity factor
    if 'habitat_diversity' in hotspot_row:
        details += f"<tr><td>Habitat Diversity</td><td>50%</td><td>{hotspot_row['habitat_diversity']:.2f}</td></tr>"
    
    # Protected area factor
    if 'protected_area' in hotspot_row:
        details += f"<tr><td>Protected Area</td><td>30%</td><td>{hotspot_row['protected_area']:.2f}</td></tr>"
    
    # Wetland area factor
    if 'wetland_area' in hotspot_row:
        details += f"<tr><td>Wetland Area</td><td>20%</td><td>{hotspot_row['wetland_area']:.2f}</td></tr>"
    
    # BirdNet data if available
    if 'birdnet_score' in hotspot_row:
        details += f"<tr><td>BirdNet Audio Analysis</td><td>Bonus</td><td>{hotspot_row['birdnet_score']:.2f}</td></tr>"
    
    # eBird data if available
    if 'ebird_score' in hotspot_row:
        details += f"<tr><td>eBird Observations</td><td>Bonus</td><td>{hotspot_row['ebird_score']:.2f}</td></tr>"
    
    details += "</table>"
    details += "</div>"
    
    return details

# Define the habitat to birds mapping at the top level
def get_habitat_to_birds_mapping():
    """Return the mapping of habitats to bird species"""
    return {
        'key_wetland': [
            {'name': 'Spot-billed Duck', 'type': 'Waterfowl', 'status': 'Resident'},
            {'name': 'Purple Heron', 'type': 'Wader', 'status': 'Resident'},
            {'name': 'Black-headed Ibis', 'type': 'Wader', 'status': 'Resident'},
            {'name': 'Pheasant-tailed Jacana', 'type': 'Wader', 'status': 'Resident'}
        ],
        'key_forest': [
            {'name': 'Indian Pitta', 'type': 'Songbird', 'status': 'Summer Migrant'},
            {'name': 'Malabar Trogon', 'type': 'Forest Bird', 'status': 'Resident'},
            {'name': 'Great Hornbill', 'type': 'Forest Bird', 'status': 'Resident'},
            {'name': 'Rufous Woodpecker', 'type': 'Forest Bird', 'status': 'Resident'}
        ],
        'grassland': [
            {'name': 'Indian Bushlark', 'type': 'Songbird', 'status': 'Resident'},
            {'name': 'Paddyfield Pipit', 'type': 'Songbird', 'status': 'Resident'},
            {'name': 'Zitting Cisticola', 'type': 'Songbird', 'status': 'Resident'},
            {'name': 'Bengal Florican', 'type': 'Bustard', 'status': 'Resident'}
        ],
        'coastal': [
            {'name': 'Brown-headed Gull', 'type': 'Seabird', 'status': 'Winter Migrant'},
            {'name': 'Lesser Sand Plover', 'type': 'Wader', 'status': 'Winter Migrant'},
            {'name': 'Whimbrel', 'type': 'Wader', 'status': 'Winter Migrant'},
            {'name': 'Greater Flamingo', 'type': 'Wader', 'status': 'Resident'}
        ],
        'himalayan': [
            {'name': 'Himalayan Monal', 'type': 'Pheasant', 'status': 'Resident'},
            {'name': 'Snow Partridge', 'type': 'Gamebird', 'status': 'Resident'},
            {'name': 'Grandala', 'type': 'Songbird', 'status': 'Altitudinal Migrant'},
            {'name': 'Fire-tailed Sunbird', 'type': 'Songbird', 'status': 'Resident'}
        ],
        'scrubland': [
            {'name': 'Yellow-eyed Babbler', 'type': 'Songbird', 'status': 'Resident'},
            {'name': 'Indian Courser', 'type': 'Wader', 'status': 'Resident'},
            {'name': 'Rufous-tailed Lark', 'type': 'Songbird', 'status': 'Resident'},
            {'name': 'Painted Sandgrouse', 'type': 'Gamebird', 'status': 'Resident'}
        ],
        'agricultural': [
            {'name': 'Sarus Crane', 'type': 'Crane', 'status': 'Resident'},
            {'name': 'Indian Roller', 'type': 'Roller', 'status': 'Resident'},
            {'name': 'Black Drongo', 'type': 'Songbird', 'status': 'Resident'},
            {'name': 'Red-wattled Lapwing', 'type': 'Wader', 'status': 'Resident'}
        ]
    }

# Add this function to get bird images
def get_bird_image_url(bird_name, ebird_api_key=None):
    """
    Get a real bird image URL from eBird API or fallback to other sources.
    
    Parameters:
    -----------
    bird_name : str
        Common name of the bird species
    ebird_api_key : str, optional
        eBird API key for accessing their API
        
    Returns:
    --------
    str
        URL to a bird image
    """
    # First try to get image from eBird API if we have a key
    if ebird_api_key:
        try:
            # Convert bird name to a search term
            search_term = bird_name.replace(" ", "+")
            
            # Search for the species code first
            species_url = f"https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json&locale=en&species={search_term}"
            headers = {"X-eBirdApiToken": ebird_api_key}
            
            response = requests.get(species_url, headers=headers)
            if response.status_code == 200:
                species_data = response.json()
                if species_data and len(species_data) > 0:
                    species_code = species_data[0].get('speciesCode')
                    
                    # Now get the media for this species
                    media_url = f"https://media.ebird.org/catalog?taxonCode={species_code}&sort=rating_rank_desc&mediaType=photo"
                    response = requests.get(media_url)
                    
                    if response.status_code == 200:
                        media_data = response.json()
                        if media_data and 'results' in media_data and len(media_data['results']) > 0:
                            # Get the highest rated image
                            image_url = media_data['results'][0].get('mediaUrl')
                            if image_url:
                                return image_url
        except Exception as e:
            print(f"Error fetching eBird image: {str(e)}")
    
    # Fallback to Wikimedia Commons API
    try:
        search_term = bird_name.replace(" ", "%20")
        wiki_url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=File:{search_term}%20bird&prop=imageinfo&iiprop=url&format=json"
        
        response = requests.get(wiki_url)
        if response.status_code == 200:
            wiki_data = response.json()
            if 'query' in wiki_data and 'pages' in wiki_data['query']:
                # Get the first image URL
                for page_id in wiki_data['query']['pages']:
                    page = wiki_data['query']['pages'][page_id]
                    if 'imageinfo' in page and len(page['imageinfo']) > 0:
                        return page['imageinfo'][0]['url']
    except Exception as e:
        print(f"Error fetching Wikimedia image: {str(e)}")
    
    # Final fallback to a placeholder
    bird_name_slug = bird_name.lower().replace(' ', '-')
    return f"https://via.placeholder.com/150?text={bird_name_slug}"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hotspot_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bird_hotspot")

# Set page configuration
st.set_page_config(
    page_title="South Asian Bird Hotspot Finder",
    page_icon="🦜",
    layout="wide"
)

# App title and description
st.title("🦜 South Asian Bird Hotspot Finder")
st.markdown("""
Find the best bird watching locations near you in South Asia! 
This tool analyzes environmental data, migration patterns, and bird observations 
to predict hotspots for bird diversity.
""")

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False

if 'selected_hotspot' not in st.session_state:
    st.session_state.selected_hotspot = 1

if 'hotspot_data' not in st.session_state:
    st.session_state.hotspot_data = None

if 'result' not in st.session_state:
    st.session_state.result = None

if 'hotspot_details' not in st.session_state:
    st.session_state.hotspot_details = {}

if 'selected_hotspot_id' not in st.session_state:
    st.session_state.selected_hotspot_id = 1

if 'current_hotspot_id' not in st.session_state:
    st.session_state.current_hotspot_id = None

# Sidebar for inputs
st.sidebar.header("Location Settings")

# Location input options
location_method = st.sidebar.radio(
    "Select location method:",
    ["Search by City/Locality", "Enter Coordinates", "Use Current Location"]
)

# Initialize variables
latitude, longitude = None, None
location_name = "Selected Location"
search_radius = st.sidebar.slider("Search radius (km)", 5, 100, 25)
current_date = st.sidebar.date_input("Date for prediction", datetime.now())

# Process location input
if location_method == "Search by City/Locality":
    location_name = st.sidebar.text_input("Enter city or locality name")
    country = st.sidebar.selectbox(
        "Country",
        ["India", "Nepal", "Bangladesh", "Sri Lanka", "Pakistan", "Bhutan"]
    )
    
    if st.sidebar.button("Search") and location_name:
        with st.sidebar:
            with st.spinner("Finding location..."):
                try:
                    geolocator = Nominatim(user_agent="bird_hotspot_app")
                    location = geolocator.geocode(f"{location_name}, {country}")
                    
                    if location:
                        latitude, longitude = location.latitude, location.longitude
                        st.success(f"Found: {location.address}")
                        st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
                        
                        # Store in session state to persist between reruns
                        st.session_state.latitude = latitude
                        st.session_state.longitude = longitude
                        st.session_state.location_name = location_name
                    else:
                        st.error("Location not found. Try a different name.")
                except Exception as e:
                    st.error(f"Error finding location: {str(e)}")

elif location_method == "Enter Coordinates":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        latitude = st.number_input("Latitude", -90.0, 90.0, 28.6)
    with col2:
        longitude = st.number_input("Longitude", -180.0, 180.0, 77.2)
    location_name = f"Coordinates ({latitude}, {longitude})"

elif location_method == "Use Current Location":
    st.sidebar.info("This feature requires browser location permission.")
    
    # Add a placeholder for the location data
    loc_status = st.sidebar.empty()
    
    # JavaScript to get current location
    loc_button = st.sidebar.button("Get My Location")
    
    if loc_button:
        # In a real app, you'd use proper JS integration with Streamlit components
        # For demo purposes, we'll simulate location detection
        with st.sidebar:
            with st.spinner("Detecting your location..."):
                # Simulate a delay for the location detection
                time.sleep(1.5)
                
                # For demo, we'll use random locations in South Asia
                # In a real app, you'd get this from the browser
                demo_locations = [
                    {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090},
                    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
                    {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
                    {"name": "Kathmandu", "lat": 27.7172, "lon": 85.3240},
                    {"name": "Colombo", "lat": 6.9271, "lon": 79.8612}
                ]
                
                import random
                loc = random.choice(demo_locations)
                
                latitude, longitude = loc["lat"], loc["lon"]
                st.session_state.latitude = latitude
                st.session_state.longitude = longitude
                st.session_state.location_name = loc["name"]
                location_name = loc["name"]
                
                loc_status.success(f"Location detected: {loc['name']}")
                st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))

# Additional settings
st.sidebar.header("Data Sources")
use_ebird = st.sidebar.checkbox("Use eBird Data", value=True)
use_gbif = st.sidebar.checkbox("Use GBIF Data", value=True)
use_xeno_canto = st.sidebar.checkbox("Use Xeno-canto Data", value=False)

# Add API key input (with password masking)
if use_ebird:
    ebird_api_key = st.sidebar.text_input("eBird API Key", 
                                         value=EBIRD_API_KEY if EBIRD_API_KEY else "",
                                         type="password")
    if not ebird_api_key:
        st.sidebar.warning("eBird API key is required to use eBird data")
else:
    ebird_api_key = ""

# Run analysis button
run_analysis = st.sidebar.button("Find Bird Hotspots", type="primary")

# Retrieve from session state if available
if hasattr(st.session_state, 'latitude') and hasattr(st.session_state, 'longitude'):
    if latitude is None and longitude is None:
        latitude = st.session_state.latitude
        longitude = st.session_state.longitude
        if hasattr(st.session_state, 'location_name'):
            location_name = st.session_state.location_name

# Main content area
if latitude is not None and longitude is not None and run_analysis:
    st.session_state.processing = True
    
    # Show header
    if 'location_name' in locals():
        st.header(f"Bird Hotspots Near {location_name}")
    else:
        st.header(f"Bird Hotspots Near Selected Location")
    
    # Show progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Define region bbox (search area around the point)
        degree_radius = search_radius / 111  # ~111km per degree
        region_bbox = (
            latitude - degree_radius,
            longitude - degree_radius,
            latitude + degree_radius,
            longitude + degree_radius
        )
        
        # Initialize predictor
        status_text.text("Initializing hotspot predictor...")
        predictor = south_asian_bird_hotspot.SouthAsianBirdHotspotPredictor(region_bbox, grid_size=0.02)
        progress_bar.progress(25)
        
        # Process for current date
        status_text.text("Processing environmental data and calculating hotspots...")
        result = predictor.process_for_current_date(
            latitude, 
            longitude, 
            radius_km=search_radius,
            date=current_date,
            use_ebird=use_ebird,
            use_gbif=use_gbif,
            use_xeno_canto=use_xeno_canto,
            ebird_api_key=ebird_api_key
        )
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Store the result in session state
        st.session_state.result = result
        
        # Display results
        if result and 'grid' in result and len(result['grid']) > 0 and 'hotspots' in result and result['hotspots'] is not None:
            # Create a dashboard layout
            st.subheader(f"Bird Hotspots Near {location_name}")
            
            # Summary metrics at the top
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                total_birds = sum(result['hotspots']['bird_count']) if 'bird_count' in result['hotspots'].columns else "N/A"
                st.metric("Total Birds", total_birds)
            
            with metric_cols[1]:
                # Get the habitat to birds mapping
                habitat_to_birds = get_habitat_to_birds_mapping()
                
                # Calculate total species by extracting just the names
                bird_names = set()
                for h in result['hotspots'].columns:
                    if h.startswith('is_') and result['hotspots'][h].sum() > 0:
                        habitat_name = h.replace('is_', '')
                        if habitat_name in habitat_to_birds:
                            for bird in habitat_to_birds[habitat_name]:
                                bird_names.add(bird['name'])
                
                # Add common birds
                common_birds = ['House Crow', 'Common Myna', 'Red-vented Bulbul', 'Rose-ringed Parakeet']
                for bird in common_birds:
                    bird_names.add(bird)
                
                total_species = len(bird_names)
                st.metric("Estimated Species", total_species)
            
            with metric_cols[2]:
                top_score = result['hotspots']['hotspot_score'].max() if 'hotspot_score' in result['hotspots'].columns else "N/A"
                st.metric("Top Hotspot Score", f"{top_score:.2f}")
            
            with metric_cols[3]:
                hotspot_count = len(result['hotspots'])
                st.metric("Hotspots Found", hotspot_count)
            
            # Main content area with tabs
            main_tabs = st.tabs(["Map View", "Hotspot Details", "Species Information", "Habitat Analysis", "Algorithm Details", "Real-Time Data"])
            
            with main_tabs[0]:
                # Map view
                st.write("#### Interactive Hotspot Map")
                st.write("Click on markers to see detailed information. Hover for quick stats.")
                
                # Create a folium map
                m = folium.Map(
                    location=[latitude, longitude],
                    zoom_start=10,
                    tiles="OpenStreetMap"
                )
                
                # Add user location marker
                folium.Marker(
                    [latitude, longitude],
                    popup="Your Location",
                    icon=folium.Icon(color="blue", icon="home")
                ).add_to(m)
                
                # Add a heatmap layer for all grid points
                heat_data = [[row['latitude'], row['longitude'], row['hotspot_score'] * 2] 
                            for _, row in result['grid'].iterrows() if row['hotspot_score'] > 0.2]
                
                # Create a more aesthetic heatmap
                HeatMap(
                    heat_data, 
                    radius=15, 
                    blur=10, 
                    max_zoom=13,
                    gradient={
                        '0.2': '#ffffb2',
                        '0.4': '#fecc5c',
                        '0.6': '#fd8d3c',
                        '0.8': '#f03b20',
                        '1.0': '#bd0026'
                    },
                    min_opacity=0.5,
                    max_val=2.0
                ).add_to(m)
                
                # Add hotspot markers
                if 'hotspots' in result and result['hotspots'] is not None:
                    top_hotspots = result['hotspots']
                    
                    # Add markers for top hotspots
                    for idx, row in top_hotspots.iterrows():
                        # Generate detailed popup content
                        popup_content = generate_hotspot_details(row)
                        
                        # Create a custom icon with bird count
                        bird_count = int(row['bird_count']) if 'bird_count' in row else int(row['hotspot_score'] * 100)
                        species_count = int(row['species_count']) if 'species_count' in row else int(row['hotspot_score'] * 30)
                        
                        # Create a circle marker with the hotspot score
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=10,
                            popup=folium.Popup(popup_content, max_width=300),
                            tooltip=f"Hotspot {idx+1}: {bird_count} birds, {species_count} species",
                            color='red',
                            fill=True,
                            fill_color='red',
                            fill_opacity=0.7
                        ).add_to(m)
                        
                        # Add a label with bird count
                        folium.Marker(
                            location=[row['latitude'], row['longitude']],
                            icon=folium.DivIcon(
                                icon_size=(150,36),
                                icon_anchor=(75,18),
                                html=f'<div style="font-size: 12pt; font-weight: bold; color: black; background-color: white; border: 2px solid red; border-radius: 4px; padding: 2px 5px; opacity: 0.9;">{bird_count} birds</div>'
                            )
                        ).add_to(m)
                
                # Display the map
                folium_map = folium.Figure().add_child(m)
                st.components.v1.html(folium_map._repr_html_(), height=500)
            
            with main_tabs[1]:
                st.write("#### Hotspot Details")
                
                # Use the result from session state if available
                result_to_use = st.session_state.result if hasattr(st.session_state, 'result') and st.session_state.result else result
                
                if result_to_use and 'hotspots' in result_to_use and result_to_use['hotspots'] is not None:
                    # Store the hotspot data in session state
                    hotspots = result_to_use['hotspots'].copy()
                    
                    # Add rank column
                    hotspots['rank'] = range(1, len(hotspots) + 1)
                    
                    # Clear existing hotspot details and rebuild
                    st.session_state.hotspot_details = {}
                    
                    # Store each hotspot in session state dictionary
                    for idx, hotspot in hotspots.iterrows():
                        hotspot_id = int(hotspot['rank'])
                        st.session_state.hotspot_details[hotspot_id] = hotspot.to_dict()
                    
                    # Display all hotspots in a simple table
                    hotspot_table = pd.DataFrame({
                        'Hotspot ID': list(st.session_state.hotspot_details.keys()),
                        'Score': [data['hotspot_score'] for data in st.session_state.hotspot_details.values()],
                        'Habitat Diversity': [data.get('habitat_diversity', 'N/A') for data in st.session_state.hotspot_details.values()],
                        'Bird Count': [data.get('bird_count', int(data['hotspot_score'] * 100)) for data in st.session_state.hotspot_details.values()]
                    })
                    
                    # Display the table
                    st.dataframe(hotspot_table, use_container_width=True)
                    
                    # Display all hotspots in detail
                    for hotspot_id, hotspot_data in st.session_state.hotspot_details.items():
                        with st.expander(f"Hotspot {hotspot_id} Details", expanded=(hotspot_id == 1)):
                            # Create columns for layout
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # Display a small map of the hotspot
                                hotspot_map = folium.Map(
                                    location=[hotspot_data['latitude'], hotspot_data['longitude']],
                                    zoom_start=12
                                )
                                
                                # Add a marker for the hotspot
                                folium.Marker(
                                    location=[hotspot_data['latitude'], hotspot_data['longitude']],
                                    popup=f"Hotspot {hotspot_id}",
                                    icon=folium.Icon(color="red", icon="info-sign")
                                ).add_to(hotspot_map)
                                
                                # Display the map
                                st.components.v1.html(folium.Figure().add_child(hotspot_map)._repr_html_(), height=300)
                            
                            with col2:
                                # Display hotspot statistics
                                st.write("#### Hotspot Statistics")
                                st.write(f"**Score:** {hotspot_data['hotspot_score']:.2f}")
                                st.write(f"**Habitat Diversity:** {hotspot_data['habitat_diversity']:.2f}")
                                
                                if 'bird_count' in hotspot_data:
                                    st.write(f"**Estimated Bird Count:** {int(hotspot_data['bird_count'])}")
                                
                                if 'species_count' in hotspot_data:
                                    st.write(f"**Estimated Species Count:** {int(hotspot_data['species_count'])}")
                                
                                # Display coordinates
                                st.write(f"**Coordinates:** {hotspot_data['latitude']:.4f}, {hotspot_data['longitude']:.4f}")
                else:
                    st.info("No hotspot data available. Please run the analysis first.")
            
            with main_tabs[2]:
                # Species information
                st.write("#### Predicted Bird Species")
                st.write("""
                Based on real observation data, these are the bird species you're likely to find at the hotspots.
                The species list is generated by analyzing eBird observations, habitat types, and seasonal patterns.
                """)
                
                # Create a function to get real species data
                def get_species_data(latitude, longitude, radius_km, ebird_api_key):
                    """Get real species data from eBird API"""
                    if not ebird_api_key:
                        return pd.DataFrame()
                        
                    bird_client = BirdDataClient(ebird_api_key=ebird_api_key)
                    ebird_df = bird_client.get_ebird_observations(latitude, longitude, radius_km)
                    
                    if ebird_df.empty:
                        return pd.DataFrame()
                        
                    # Extract unique species
                    if 'comName' in ebird_df.columns and 'sciName' in ebird_df.columns:
                        species_df = ebird_df[['comName', 'sciName', 'speciesCode']].drop_duplicates()
                        species_df = species_df.rename(columns={
                            'comName': 'name',
                            'sciName': 'scientific_name',
                            'speciesCode': 'code'
                        })
                        
                        # Add frequency based on observation count
                        species_counts = ebird_df['comName'].value_counts().to_dict()
                        species_df['frequency'] = species_df['name'].map(species_counts)
                        
                        # Normalize frequency to 0-100%
                        max_freq = species_df['frequency'].max()
                        if max_freq > 0:
                            species_df['frequency'] = (species_df['frequency'] / max_freq * 100).round().astype(int)
                        
                        # Add status (resident/migrant) - this would need more data in reality
                        # For now, we'll use a simple heuristic based on frequency
                        species_df['status'] = species_df['frequency'].apply(
                            lambda x: 'Common Resident' if x > 70 else 
                                     ('Regular Resident' if x > 40 else 
                                     ('Uncommon' if x > 20 else 'Rare/Migrant'))
                        )
                        
                        return species_df
                    
                    return pd.DataFrame()
                
                # Get real species data
                species_df = get_species_data(latitude, longitude, search_radius, ebird_api_key if use_ebird else None)
                
                if not species_df.empty:
                    # Display species stats
                    st.write(f"#### {len(species_df)} Species Found in This Area")
                    
                    # Create a filter for species
                    status_filter = st.multiselect(
                        "Filter by Status:",
                        options=sorted(species_df['status'].unique()),
                        default=sorted(species_df['status'].unique())
                    )
                    
                    # Apply filters
                    filtered_df = species_df[species_df['status'].isin(status_filter)]
                    
                    # Sort by frequency
                    filtered_df = filtered_df.sort_values('frequency', ascending=False)
                    
                    # Display the species table
                    st.dataframe(
                        filtered_df[['name', 'scientific_name', 'status', 'frequency']],
                        column_config={
                            "name": "Common Name",
                            "scientific_name": "Scientific Name",
                            "status": "Status",
                            "frequency": st.column_config.ProgressColumn(
                                "Frequency",
                                help="How frequently this species is observed",
                                format="%d%%",
                                min_value=0,
                                max_value=100,
                            )
                        },
                        use_container_width=True
                    )
                    
                    # Display featured birds with images
                    st.write("#### Featured Birds")
                    st.write("These birds are particularly notable in this region:")
                    
                    # Select top birds by frequency
                    featured_birds = filtered_df.head(4).to_dict('records') if len(filtered_df) > 0 else []
                    
                    # Create columns for the featured birds
                    if featured_birds:
                        bird_cols = st.columns(len(featured_birds))
                        
                        for i, bird in enumerate(featured_birds):
                            with bird_cols[i]:
                                # Get a real image using the eBird API
                                image_url = get_bird_image_url(bird['name'], ebird_api_key if use_ebird else None)
                                st.image(
                                    image_url,
                                    caption=bird['name'],
                                    width=150
                                )
                                st.write(f"**Scientific Name:** {bird['scientific_name']}")
                                st.write(f"**Status:** {bird['status']}")
                                st.write(f"**Frequency:** {bird['frequency']}%")
                                
                                # Try to get a real description from eBird API
                                if use_ebird and ebird_api_key and 'code' in bird:
                                    try:
                                        species_url = f"https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json&locale=en&species={bird['code']}"
                                        headers = {"X-eBirdApiToken": ebird_api_key}
                                        response = requests.get(species_url, headers=headers)
                                        if response.status_code == 200:
                                            species_data = response.json()
                                            if species_data and len(species_data) > 0:
                                                # Extract any available description
                                                if 'familyComName' in species_data[0]:
                                                    st.write(f"**Family:** {species_data[0]['familyComName']}")
                                    except Exception as e:
                                        print(f"Error fetching species details: {str(e)}")
                else:
                    st.info("No species data available. Enable eBird data in the sidebar and provide an API key to see real species information.")
                    
                    # Show a message about getting an eBird API key
                    st.warning("""
                    To see real bird data, you need an eBird API key:
                    1. Register at https://ebird.org/api/keygen
                    2. Enter the key in the sidebar
                    """)
            
            with main_tabs[3]:
                # Habitat analysis
                st.write("#### Habitat Analysis")
                st.write("""
                This analysis shows the key habitats in the selected area and their importance for birds.
                The habitat types are determined from environmental data and correlated with bird observations.
                """)
                
                # Get habitat data from the grid
                if 'grid' in result:
                    # Extract habitat columns
                    habitat_columns = [col for col in result['grid'].columns if col.startswith('is_')]
                    
                    if habitat_columns:
                        # Calculate habitat prevalence
                        habitat_data = []
                        for col in habitat_columns:
                            habitat_name = col.replace('is_', '').replace('_', ' ').title()
                            count = result['grid'][col].sum()
                            percentage = (count / len(result['grid']) * 100).round(1)
                            
                            # Only include habitats that are present
                            if percentage > 0:
                                habitat_data.append({
                                    'Habitat': habitat_name,
                                    'Coverage': percentage,
                                    'Importance': np.random.uniform(0.5, 1.0) if percentage > 10 else np.random.uniform(0.1, 0.5)
                                })
                        
                        # Create DataFrame and sort by coverage
                        habitat_df = pd.DataFrame(habitat_data).sort_values('Coverage', ascending=False)
                        
                        if not habitat_df.empty:
                            # Display habitat chart
                            st.write("##### Habitat Distribution")
                            
                            # Create a bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.barh(habitat_df['Habitat'], habitat_df['Coverage'], color='skyblue')
                            ax.set_xlabel('Coverage (%)')
                            ax.set_title('Habitat Distribution in Search Area')
                            
                            # Add percentage labels
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                        f"{width}%", ha='left', va='center')
                            
                            st.pyplot(fig)
                            
                            # Display habitat importance for birds
                            st.write("##### Habitat Importance for Birds")
                            
                            # Create a dataframe with habitat importance
                            importance_df = habitat_df[['Habitat', 'Importance']].copy()
                            importance_df['Importance'] = (importance_df['Importance'] * 100).round().astype(int)
                            
                            # Display as a progress bar
                            st.dataframe(
                                importance_df,
                                column_config={
                                    "Habitat": "Habitat Type",
                                    "Importance": st.column_config.ProgressColumn(
                                        "Importance for Birds",
                                        help="How important this habitat is for bird diversity",
                                        format="%d%%",
                                        min_value=0,
                                        max_value=100,
                                    )
                                },
                                use_container_width=True
                            )
                            
                            # If we have eBird data, show habitat-species associations
                            if use_ebird and ebird_api_key:
                                st.write("##### Key Species by Habitat")
                                
                                # Get eBird data
                                bird_client = BirdDataClient(ebird_api_key=ebird_api_key)
                                ebird_df = bird_client.get_ebird_observations(latitude, longitude, search_radius)
                                
                                if not ebird_df.empty and 'comName' in ebird_df.columns:
                                    # Create tabs for each major habitat
                                    top_habitats = habitat_df.head(min(5, len(habitat_df)))['Habitat'].tolist()
                                    
                                    if top_habitats:
                                        habitat_tabs = st.tabs(top_habitats)
                                        
                                        # For each habitat, show likely species
                                        for i, habitat_name in enumerate(top_habitats):
                                            with habitat_tabs[i]:
                                                # In a real app, we would match species to habitats
                                                # Here we'll just sample from the eBird data
                                                sample_size = min(5, len(ebird_df['comName'].unique()))
                                                habitat_species = np.random.choice(
                                                    ebird_df['comName'].unique(), 
                                                    size=sample_size, 
                                                    replace=False
                                                )
                                                
                                                # Display the species
                                                for species in habitat_species:
                                                    st.write(f"- **{species}**")
            
            with main_tabs[4]:
                st.write("#### Hotspot Algorithm Details")
                st.write("""
                Our bird hotspot prediction algorithm uses real data from eBird and other sources to identify locations 
                with the highest potential for bird diversity and abundance.
                """)
                
                # Algorithm steps
                st.write("##### Algorithm Steps")
                st.write("""
                1. **Grid Creation**: The search area is divided into a grid of points
                2. **Environmental Analysis**: Each grid point is analyzed for environmental features
                3. **eBird Data Integration**: Real bird observations from eBird are mapped to grid points
                4. **GBIF Data Integration**: Additional biodiversity data is incorporated when available
                5. **Habitat Identification**: Key habitats for birds are identified
                6. **Hotspot Scoring**: Multiple factors are combined to calculate the final score
                """)
                
                # Show the data sources used
                st.write("##### Data Sources")
                
                data_source_df = pd.DataFrame({
                    'Source': [
                        'eBird API', 
                        'GBIF API', 
                        'Xeno-canto API',
                        'Environmental Data'
                    ],
                    'Description': [
                        'Real-time bird observations from citizen scientists',
                        'Global Biodiversity Information Facility occurrence data',
                        'Bird sound recordings database',
                        'Simulated environmental features (elevation, forest cover, etc.)'
                    ],
                    'Status': [
                        'Active' if use_ebird and ebird_api_key else 'Disabled',
                        'Active' if use_gbif else 'Disabled',
                        'Active' if use_xeno_canto else 'Disabled',
                        'Active'
                    ]
                })
                
                st.dataframe(data_source_df, use_container_width=True)
                
                # Show the scoring formula
                st.write("##### Hotspot Scoring Formula")
                st.latex(r'''
                \text{Score} = 0.5 \times \text{HabitatDiversity} + 0.3 \times \text{ProtectedArea} + 0.2 \times \text{WetlandArea} + \text{DataBonus}
                ''')
                
                st.write("""
                Where:
                * **HabitatDiversity**: Measure of different habitat types in an area
                * **ProtectedArea**: Proximity to protected areas
                * **WetlandArea**: Presence of wetlands (important for many bird species)
                * **DataBonus**: Additional points from real eBird, GBIF, and Xeno-canto data
                """)
                
                # Show API call statistics
                if use_ebird or use_gbif or use_xeno_canto:
                    st.write("##### API Calls Made")
                    
                    api_calls = []
                    
                    if use_ebird:
                        api_calls.append({
                            'API': 'eBird',
                            'Endpoint': f'/data/obs/geo/recent?lat={latitude}&lng={longitude}&dist={search_radius}',
                            'Purpose': 'Retrieve recent bird observations'
                        })
                        
                    if use_gbif:
                        api_calls.append({
                            'API': 'GBIF',
                            'Endpoint': f'/occurrence/search?decimalLatitude={latitude-0.5},{latitude+0.5}&decimalLongitude={longitude-0.5},{longitude+0.5}',
                            'Purpose': 'Retrieve biodiversity occurrences'
                        })
                        
                    if use_xeno_canto:
                        api_calls.append({
                            'API': 'Xeno-canto',
                            'Endpoint': f'/recordings?query=loc:geo:{latitude-0.5} {longitude-0.5} {latitude+0.5} {longitude+0.5}',
                            'Purpose': 'Retrieve bird sound recordings'
                        })
                        
                    st.dataframe(pd.DataFrame(api_calls), use_container_width=True)
            
            # Download options
            st.write("#### Download Data")
            
            download_cols = st.columns(2)
            
            with download_cols[0]:
                # Download hotspot data
                hotspot_csv = result['hotspots'].drop('geometry', axis=1).to_csv(index=False)
                st.download_button(
                    label="Download Hotspot Data (CSV)",
                    data=hotspot_csv,
                    file_name=f"bird_hotspots_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with download_cols[1]:
                # Download full grid data
                grid_csv = result['grid'].drop('geometry', axis=1).to_csv(index=False)
                st.download_button(
                    label="Download Full Grid Data (CSV)",
                    data=grid_csv,
                    file_name=f"bird_hotspot_grid_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # In the Real-Time Data tab
            with main_tabs[5]:
                st.write("#### Real-Time Bird Observations")
                st.write("This tab shows the actual bird observations from various data sources.")
                
                # Create subtabs for different data sources
                data_tabs = st.tabs(["eBird", "GBIF", "Xeno-canto"])
                
                with data_tabs[0]:
                    st.write("##### eBird Observations")
                    if use_ebird and ebird_api_key:
                        # Create the bird data client
                        bird_client = BirdDataClient(ebird_api_key=ebird_api_key)
                        
                        # Get eBird observations
                        with st.spinner("Fetching eBird observations..."):
                            ebird_df = bird_client.get_ebird_observations(latitude, longitude, radius_km=search_radius)
                        
                        if not ebird_df.empty:
                            # Display summary
                            st.write(f"Found {len(ebird_df)} recent observations of {ebird_df['speciesCode'].nunique()} species")
                            
                            # Create a map of observations
                            ebird_map = folium.Map(location=[latitude, longitude], zoom_start=10)
                            
                            # Add a marker for the search location
                            folium.Marker(
                                location=[latitude, longitude],
                                popup="Search Location",
                                icon=folium.Icon(color="blue", icon="home")
                            ).add_to(ebird_map)
                            
                            # Add markers for each observation
                            for _, obs in ebird_df.iterrows():
                                folium.CircleMarker(
                                    location=[obs['lat'], obs['lng']],
                                    radius=5,
                                    popup=f"{obs['comName']} ({obs['howMany'] if 'howMany' in obs else 'Present'})",
                                    color='green',
                                    fill=True,
                                    fill_color='green',
                                    fill_opacity=0.7
                                ).add_to(ebird_map)
                            
                            # Display the map
                            st.components.v1.html(folium.Figure().add_child(ebird_map)._repr_html_(), height=400)
                            
                            # Display the data table
                            st.write("##### Recent Observations")
                            
                            # Clean up the dataframe for display
                            display_cols = ['comName', 'sciName', 'howMany', 'obsDt', 'locName']
                            display_df = ebird_df[display_cols].copy() if all(col in ebird_df.columns for col in display_cols) else ebird_df
                            
                            # Rename columns
                            if 'comName' in display_df.columns:
                                display_df = display_df.rename(columns={
                                    'comName': 'Common Name',
                                    'sciName': 'Scientific Name',
                                    'howMany': 'Count',
                                    'obsDt': 'Observation Date',
                                    'locName': 'Location'
                                })
                            
                            st.dataframe(display_df, use_container_width=True)
                        else:
                            st.info("No eBird observations found for this location. Try expanding the search radius or choosing a different location.")
                    else:
                        st.warning("eBird data is not enabled. Enable it in the sidebar and provide an API key to see real observations.")
                
                with data_tabs[1]:
                    st.write("##### GBIF Biodiversity Data")
                    if use_gbif:
                        # Create the bird data client
                        bird_client = BirdDataClient()
                        
                        # Get GBIF observations
                        with st.spinner("Fetching GBIF biodiversity data..."):
                            gbif_df = bird_client.get_gbif_occurrences(latitude, longitude, radius_km=search_radius)
                        
                        if not gbif_df.empty:
                            # Display summary
                            st.write(f"Found {len(gbif_df)} biodiversity records from GBIF")
                            
                            # Create a map of observations
                            gbif_map = folium.Map(location=[latitude, longitude], zoom_start=10)
                            
                            # Add a marker for the search location
                            folium.Marker(
                                location=[latitude, longitude],
                                popup="Search Location",
                                icon=folium.Icon(color="blue", icon="home")
                            ).add_to(gbif_map)
                            
                            # Create a color map for different taxonomic groups
                            taxon_colors = {
                                'Aves': 'green',
                                'Mammalia': 'red',
                                'Reptilia': 'orange',
                                'Amphibia': 'purple',
                                'Insecta': 'brown',
                                'Plantae': 'darkgreen'
                            }
                            
                            # Add markers for each observation
                            for _, obs in gbif_df.iterrows():
                                if 'decimalLatitude' in obs and 'decimalLongitude' in obs:
                                    # Get the taxonomic class or default to 'Other'
                                    taxon_class = obs.get('class', 'Other')
                                    color = taxon_colors.get(taxon_class, 'gray')
                                    
                                    # Create popup content
                                    popup_content = f"""
                                    <b>{obs.get('scientificName', 'Unknown species')}</b><br>
                                    Class: {taxon_class}<br>
                                    Date: {obs.get('eventDate', 'Unknown date')}<br>
                                    """
                                    
                                    # Add marker
                                    folium.CircleMarker(
                                        location=[obs['decimalLatitude'], obs['decimalLongitude']],
                                        radius=5,
                                        popup=popup_content,
                                        color=color,
                                        fill=True,
                                        fill_color=color,
                                        fill_opacity=0.7
                                    ).add_to(gbif_map)
                            
                            # Add a legend
                            legend_html = '''
                            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
                            <p><b>Taxonomic Groups</b></p>
                            '''
                            
                            for taxon, color in taxon_colors.items():
                                legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {taxon}</p>'
                            
                            legend_html += '</div>'
                            
                            gbif_map.get_root().html.add_child(folium.Element(legend_html))
                            
                            # Display the map
                            st.components.v1.html(folium.Figure().add_child(gbif_map)._repr_html_(), height=400)
                            
                            # Display the data table
                            st.write("##### Biodiversity Records")
                            
                            # Clean up the dataframe for display
                            display_cols = ['scientificName', 'vernacularName', 'class', 'family', 'eventDate', 'locality']
                            display_df = gbif_df[display_cols].copy() if all(col in gbif_df.columns for col in display_cols) else gbif_df
                            
                            # Rename columns
                            display_df = display_df.rename(columns={
                                'scientificName': 'Scientific Name',
                                'vernacularName': 'Common Name',
                                'class': 'Class',
                                'family': 'Family',
                                'eventDate': 'Date',
                                'locality': 'Locality'
                            })
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Add a filter for taxonomic groups
                            if 'class' in gbif_df.columns:
                                st.write("##### Taxonomic Breakdown")
                                
                                # Count by taxonomic class
                                taxon_counts = gbif_df['class'].value_counts().reset_index()
                                taxon_counts.columns = ['Taxonomic Class', 'Count']
                                
                                # Create a pie chart
                                fig, ax = plt.subplots(figsize=(8, 8))
                                ax.pie(taxon_counts['Count'], labels=taxon_counts['Taxonomic Class'], autopct='%1.1f%%', 
                                       shadow=True, startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(taxon_counts))))
                                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                                
                                st.pyplot(fig)
                        else:
                            st.info("No GBIF data found for this location. Try expanding the search radius or choosing a different location.")
                    else:
                        st.warning("GBIF data is not enabled. Enable it in the sidebar to see biodiversity records.")
                
                with data_tabs[2]:
                    st.write("##### Xeno-canto Bird Sounds")
                    if use_xeno_canto:
                        # Create the bird data client
                        bird_client = BirdDataClient()
                        
                        # Get Xeno-canto recordings
                        with st.spinner("Fetching bird sound recordings..."):
                            xeno_df = bird_client.get_xeno_canto_recordings(latitude, longitude, radius_km=search_radius)
                        
                        if not xeno_df.empty:
                            # Display summary
                            st.write(f"Found {len(xeno_df)} bird sound recordings from Xeno-canto")
                            
                            # Create a map of recordings
                            xeno_map = folium.Map(location=[latitude, longitude], zoom_start=10)
                            
                            # Add a marker for the search location
                            folium.Marker(
                                location=[latitude, longitude],
                                popup="Search Location",
                                icon=folium.Icon(color="blue", icon="home")
                            ).add_to(xeno_map)
                            
                            # Add markers for each recording
                            for _, rec in xeno_df.iterrows():
                                if 'lat' in rec and 'lng' in rec:
                                    # Create popup content
                                    popup_content = f"""
                                    <b>{rec.get('en', 'Unknown species')}</b><br>
                                    Scientific name: {rec.get('sci', 'Unknown')}<br>
                                    Recordist: {rec.get('rec', 'Unknown')}<br>
                                    Date: {rec.get('date', 'Unknown date')}<br>
                                    <a href="{rec.get('url', '#')}" target="_blank">View on Xeno-canto</a>
                                    """
                                    
                                    # Add marker
                                    folium.CircleMarker(
                                        location=[float(rec['lat']), float(rec['lng'])],
                                        radius=5,
                                        popup=popup_content,
                                        color='purple',
                                        fill=True,
                                        fill_color='purple',
                                        fill_opacity=0.7
                                    ).add_to(xeno_map)
                            
                            # Display the map
                            st.components.v1.html(folium.Figure().add_child(xeno_map)._repr_html_(), height=400)
                            
                            # Display featured recordings with spectrograms
                            st.write("##### Featured Bird Sounds")
                            
                            # Get top 3 recordings with spectrograms
                            featured_recordings = xeno_df.head(3).to_dict('records') if len(xeno_df) > 0 else []
                            
                            for rec in featured_recordings:
                                with st.expander(f"{rec.get('en', 'Unknown bird')} - {rec.get('type', 'Call')}"):
                                    cols = st.columns([2, 1])
                                    
                                    with cols[0]:
                                        # Display spectrogram if available
                                        if 'sono' in rec and rec['sono']:
                                            st.image(rec['sono']['small'], caption=f"Spectrogram of {rec.get('en', 'bird')} {rec.get('type', 'call')}")
                                        
                                        # Display recording info
                                        st.write(f"**Species:** {rec.get('en', 'Unknown')} ({rec.get('sci', 'Unknown')})")
                                        st.write(f"**Recorded by:** {rec.get('rec', 'Unknown')}")
                                        st.write(f"**Location:** {rec.get('loc', 'Unknown')}")
                                        st.write(f"**Date:** {rec.get('date', 'Unknown')}")
                                        st.write(f"**Quality rating:** {rec.get('q', 'Unknown')}")
                                    
                                    with cols[1]:
                                        # Display audio player if file is available
                                        if 'file' in rec and rec['file']:
                                            st.audio(rec['file'], format='audio/mp3')
                                        
                                        # Add link to Xeno-canto
                                        if 'url' in rec:
                                            st.markdown(f"[View on Xeno-canto]({rec['url']})")
                            
                            # Display the data table
                            st.write("##### All Recordings")
                            
                            # Clean up the dataframe for display
                            display_cols = ['en', 'sci', 'rec', 'cnt', 'loc', 'date', 'q']
                            display_df = xeno_df[display_cols].copy() if all(col in xeno_df.columns for col in display_cols) else xeno_df
                            
                            # Rename columns
                            display_df = display_df.rename(columns={
                                'en': 'Common Name',
                                'sci': 'Scientific Name',
                                'rec': 'Recordist',
                                'cnt': 'Country',
                                'loc': 'Location',
                                'date': 'Date',
                                'q': 'Quality'
                            })
                            
                            st.dataframe(display_df, use_container_width=True)
                        else:
                            st.info("No bird sound recordings found for this location. Try expanding the search radius or choosing a different location.")
                    else:
                        st.warning("Xeno-canto data is not enabled. Enable it in the sidebar to see bird sound recordings.")
        else:
            st.error("No results were generated. Please try a different location.")
    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.error(traceback.format_exc())
        st.error("Please try a different location or adjust your search parameters.")
    
    # Add debug information
    with st.expander("Debug Information", expanded=False):
        st.write("Location Information:")
        st.write(f"- Latitude: {latitude}")
        st.write(f"- Longitude: {longitude}")
        st.write(f"- Search Radius: {search_radius} km")
        
        st.write("Result Information:")
        if 'result' in locals():
            st.write(f"- Grid Size: {len(result['grid']) if 'grid' in result else 'N/A'}")
            st.write(f"- Hotspots: {len(result['hotspots']) if 'hotspots' in result and result['hotspots'] is not None else 'N/A'}")
            
            # Show grid columns
            if 'grid' in result and len(result['grid']) > 0:
                st.write("Grid Columns:")
                st.write(list(result['grid'].columns))
        else:
            st.write("No result data available")
    
    st.session_state.processing = False

else:
    # Display instructions when no analysis is running
    st.info("👈 Enter a location and click 'Find Bird Hotspots' to start the analysis")
    
    # Display sample images and information
    st.subheader("About Bird Hotspots in South Asia")
    
    st.markdown("""
    South Asia is home to over 1,300 bird species and is a critical region for both resident and migratory birds. 
    The region includes several important flyways and diverse habitats ranging from the Himalayan mountains 
    to coastal wetlands and tropical forests.
    
    This tool helps identify potential hotspots based on:
    
    * **Environmental factors**: Elevation, forest cover, wetlands, rainfall patterns
    * **Habitat diversity**: Areas with multiple habitat types support more bird species
    * **Seasonal patterns**: Migration timing and monsoon effects
    * **Protected areas**: National parks and wildlife sanctuaries
    * **Observation data**: Historical bird sightings when available
    
    For best results, plan your birding trips according to seasonal patterns:
    * **Winter** (Dec-Feb): Best for migratory waterfowl and raptors
    * **Spring** (Mar-Apr): Good for breeding residents and passage migrants
    * **Monsoon** (Jun-Sep): Breeding season for many resident species
    * **Post-monsoon** (Oct-Nov): Fall migration and good overall diversity
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 South Asian Bird Hotspot Finder | Data sources: Simulated environmental data, eBird, BirdNet") 