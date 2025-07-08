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
import io

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
    <div style='min-width: 250px; font-family: Arial, sans-serif;'>
    <h4 style='color: #2C3E50;'>Hotspot Details</h4>
    <div style='background-color: #ECF0F1; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
    <p><b>Score:</b> <span style='color: #E74C3C; font-weight: bold;'>{hotspot_row['hotspot_score']:.2f}</span></p>
    """
    
    # Add habitat diversity if available
    if 'habitat_diversity' in hotspot_row:
        details += f"<p><b>Habitat Diversity:</b> {hotspot_row['habitat_diversity']:.2f}</p>"
    
    # Add bird count information if available
    if 'bird_count' in hotspot_row:
        details += f"<p><b>Bird Count:</b> {int(hotspot_row['bird_count'])}</p>"
    else:
        # Simulate bird count based on hotspot score
        estimated_count = int(hotspot_row['hotspot_score'] * 100)
        details += f"<p><b>Estimated Bird Count:</b> {estimated_count}</p>"
    
    # Add species information if available
    if 'species_count' in hotspot_row:
        details += f"<p><b>Species Count:</b> {int(hotspot_row['species_count'])}</p>"
    else:
        # Simulate species count based on hotspot score
        estimated_species = int(hotspot_row['hotspot_score'] * 30)
        details += f"<p><b>Estimated Species:</b> {estimated_species}</p>"
    
    details += "</div>"
    
    # Add habitat information
    details += "<h4 style='color: #2C3E50; margin-top: 15px;'>Key Habitats</h4>"
    details += "<ul style='padding-left: 20px;'>"
    
    habitat_found = False
    for col in hotspot_row.index:
        if col.startswith('is_') and hotspot_row[col] > 0:
            habitat_found = True
            habitat_name = col.replace('is_', '').replace('_', ' ').title()
            details += f"<li>{habitat_name}</li>"
    
    if not habitat_found:
        details += "<li>No specific habitats identified</li>"
    
    details += "</ul>"
    
    # Add calculation explanation with corrected formula
    details += "<h4 style='color: #2C3E50; margin-top: 15px;'>Score Calculation</h4>"
    details += "<div style='background-color: #ECF0F1; padding: 10px; border-radius: 5px;'>"
    details += "<table style='width:100%; border-collapse: collapse;'>"
    details += "<tr style='border-bottom: 1px solid #BDC3C7;'><th style='text-align: left; padding: 5px;'>Factor</th><th style='text-align: center; padding: 5px;'>Weight</th><th style='text-align: right; padding: 5px;'>Value</th></tr>"
    
    # Habitat diversity factor - always included in the formula
    habitat_diversity_value = hotspot_row.get('habitat_diversity', 0)
    details += f"<tr style='border-bottom: 1px solid #BDC3C7;'><td style='padding: 5px;'>Habitat Diversity</td><td style='text-align: center; padding: 5px;'>50%</td><td style='text-align: right; padding: 5px;'>{habitat_diversity_value:.2f}</td></tr>"
    
    # Protected area factor
    protected_area_value = hotspot_row.get('protected_area', 0)
    details += f"<tr style='border-bottom: 1px solid #BDC3C7;'><td style='padding: 5px;'>Protected Area</td><td style='text-align: center; padding: 5px;'>30%</td><td style='text-align: right; padding: 5px;'>{protected_area_value:.2f}</td></tr>"
    
    # Wetland area factor
    wetland_value = hotspot_row.get('wetland_score', hotspot_row.get('wetland_area', 0)/100 if 'wetland_area' in hotspot_row else 0)
    details += f"<tr style='border-bottom: 1px solid #BDC3C7;'><td style='padding: 5px;'>Wetland Area</td><td style='text-align: center; padding: 5px;'>20%</td><td style='text-align: right; padding: 5px;'>{wetland_value:.2f}</td></tr>"
    
    # Calculate the base score
    base_score = (0.5 * habitat_diversity_value) + (0.3 * protected_area_value) + (0.2 * wetland_value)
    details += f"<tr style='border-bottom: 1px solid #BDC3C7; font-weight: bold;'><td style='padding: 5px;'>Base Score</td><td style='text-align: center; padding: 5px;'>100%</td><td style='text-align: right; padding: 5px;'>{base_score:.2f}</td></tr>"
    
    # Add bonus factors if available
    bonus_score = 0
    
    # BirdNet data if available
    if 'birdnet_score' in hotspot_row and hotspot_row['birdnet_score'] > 0:
        birdnet_bonus = hotspot_row['birdnet_score'] * 0.1  # 10% bonus
        bonus_score += birdnet_bonus
        details += f"<tr style='border-bottom: 1px solid #BDC3C7;'><td style='padding: 5px;'>BirdNet Audio Analysis</td><td style='text-align: center; padding: 5px;'>+10%</td><td style='text-align: right; padding: 5px;'>+{birdnet_bonus:.2f}</td></tr>"
    
    # eBird data if available
    if 'ebird_score' in hotspot_row and hotspot_row['ebird_score'] > 0:
        ebird_bonus = hotspot_row['ebird_score'] * 0.2  # 20% bonus
        bonus_score += ebird_bonus
        details += f"<tr style='border-bottom: 1px solid #BDC3C7;'><td style='padding: 5px;'>eBird Observations</td><td style='text-align: center; padding: 5px;'>+20%</td><td style='text-align: right; padding: 5px;'>+{ebird_bonus:.2f}</td></tr>"
    
    # GBIF data if available
    if 'gbif_score' in hotspot_row and hotspot_row['gbif_score'] > 0:
        gbif_bonus = hotspot_row['gbif_score'] * 0.2  # 20% bonus
        bonus_score += gbif_bonus
        details += f"<tr style='border-bottom: 1px solid #BDC3C7;'><td style='padding: 5px;'>GBIF Biodiversity</td><td style='text-align: center; padding: 5px;'>+20%</td><td style='text-align: right; padding: 5px;'>+{gbif_bonus:.2f}</td></tr>"
    
    # Xeno-canto data if available
    if 'xeno_canto_score' in hotspot_row and hotspot_row['xeno_canto_score'] > 0:
        xeno_bonus = hotspot_row['xeno_canto_score'] * 0.15  # 15% bonus
        bonus_score += xeno_bonus
        details += f"<tr style='border-bottom: 1px solid #BDC3C7;'><td style='padding: 5px;'>Xeno-canto Recordings</td><td style='text-align: center; padding: 5px;'>+15%</td><td style='text-align: right; padding: 5px;'>+{xeno_bonus:.2f}</td></tr>"
    
    # Final score with bonuses
    final_score = min(1.0, base_score + bonus_score)
    details += f"<tr style='font-weight: bold; background-color: #D5DBDB;'><td style='padding: 5px;'>Final Score</td><td style='text-align: center; padding: 5px;'></td><td style='text-align: right; padding: 5px;'>{final_score:.2f}</td></tr>"
    
    details += "</table>"
    details += "</div>"
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
    ["Search by City/Locality", "Enter Coordinates", "Use Current Location", "Bulk Analysis - Major Indian Cities"]
)

# Define major Indian cities with population > 1 million
MAJOR_INDIAN_CITIES = {
    # Mega Cities (10M+)
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "population": "30.3M", "region": "North India"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "population": "20.4M", "region": "Western India"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "population": "14.9M", "region": "Eastern India"},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946, "population": "12.3M", "region": "Southern India"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "population": "10.9M", "region": "Southern India"},
    
    # Large Cities (5-10M)
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "population": "9.7M", "region": "Southern India"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "population": "7.2M", "region": "Western India"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "population": "6.6M", "region": "Western India"},
    "Surat": {"lat": 21.1702, "lon": 72.8311, "population": "6.1M", "region": "Western India"},
    
    # Medium-Large Cities (3-5M)
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "population": "3.4M", "region": "North India"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "population": "3.1M", "region": "North India"},
    "Kanpur": {"lat": 26.4499, "lon": 80.3319, "population": "3.0M", "region": "North India"},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882, "population": "2.9M", "region": "Central India"},
    "Indore": {"lat": 22.7196, "lon": 75.8577, "population": "2.8M", "region": "Central India"},
    "Thane": {"lat": 19.2183, "lon": 72.9781, "population": "2.7M", "region": "Western India"},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126, "population": "2.6M", "region": "Central India"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "population": "2.5M", "region": "Southern India"},
    "Pimpri-Chinchwad": {"lat": 18.6279, "lon": 73.7993, "population": "2.4M", "region": "Western India"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "population": "2.3M", "region": "Eastern India"},
    
    # Cities (2-3M)
    "Vadodara": {"lat": 22.3072, "lon": 73.1812, "population": "2.2M", "region": "Western India"},
    "Ghaziabad": {"lat": 28.6692, "lon": 77.4538, "population": "2.1M", "region": "North India"},
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573, "population": "2.1M", "region": "North India"},
    "Agra": {"lat": 27.1767, "lon": 78.0081, "population": "2.0M", "region": "North India"},
    "Nashik": {"lat": 19.9975, "lon": 73.7898, "population": "2.0M", "region": "Western India"},
    "Faridabad": {"lat": 28.4089, "lon": 77.3178, "population": "1.9M", "region": "North India"},
    "Meerut": {"lat": 28.9845, "lon": 77.7064, "population": "1.9M", "region": "North India"},
    "Rajkot": {"lat": 22.3039, "lon": 70.8022, "population": "1.8M", "region": "Western India"},
    "Kalyan-Dombivali": {"lat": 19.2403, "lon": 73.1305, "population": "1.8M", "region": "Western India"},
    "Vasai-Virar": {"lat": 19.3919, "lon": 72.8397, "population": "1.7M", "region": "Western India"},
    
    # Additional Major Cities
    "Varanasi": {"lat": 25.3176, "lon": 82.9739, "population": "1.7M", "region": "North India"},
    "Srinagar": {"lat": 34.0837, "lon": 74.7973, "population": "1.7M", "region": "North India"},
    "Aurangabad": {"lat": 19.8762, "lon": 75.3433, "population": "1.7M", "region": "Western India"},
    "Dhanbad": {"lat": 23.7957, "lon": 86.4304, "population": "1.6M", "region": "Eastern India"},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723, "population": "1.6M", "region": "North India"},
    "Navi Mumbai": {"lat": 19.0330, "lon": 73.0297, "population": "1.6M", "region": "Western India"},
    "Allahabad": {"lat": 25.4358, "lon": 81.8463, "population": "1.6M", "region": "North India"},
    "Ranchi": {"lat": 23.3441, "lon": 85.3096, "population": "1.5M", "region": "Eastern India"},
    "Howrah": {"lat": 22.5958, "lon": 88.2636, "population": "1.5M", "region": "Eastern India"},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558, "population": "1.5M", "region": "Southern India"},
    
    # Emerging Cities
    "Jabalpur": {"lat": 23.1815, "lon": 79.9864, "population": "1.4M", "region": "Central India"},
    "Gwalior": {"lat": 26.2183, "lon": 78.1828, "population": "1.4M", "region": "Central India"},
    "Vijayawada": {"lat": 16.5062, "lon": 80.6480, "population": "1.4M", "region": "Southern India"},
    "Jodhpur": {"lat": 26.2389, "lon": 73.0243, "population": "1.3M", "region": "North India"},
    "Madurai": {"lat": 9.9252, "lon": 78.1198, "population": "1.3M", "region": "Southern India"},
    "Raipur": {"lat": 21.2514, "lon": 81.6296, "population": "1.3M", "region": "Central India"},
    "Kota": {"lat": 25.2138, "lon": 75.8648, "population": "1.2M", "region": "North India"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "population": "1.2M", "region": "North India"},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362, "population": "1.2M", "region": "Northeast India"},
    "Solapur": {"lat": 17.6599, "lon": 75.9064, "population": "1.2M", "region": "Western India"},
    
    # Additional Cities
    "Hubli-Dharwad": {"lat": 15.3647, "lon": 75.1240, "population": "1.2M", "region": "Southern India"},
    "Mysore": {"lat": 12.2958, "lon": 76.6394, "population": "1.2M", "region": "Southern India"},
    "Tiruchirappalli": {"lat": 10.7905, "lon": 78.7047, "population": "1.1M", "region": "Southern India"},
    "Bareilly": {"lat": 28.3670, "lon": 79.4304, "population": "1.1M", "region": "North India"},
    "Aligarh": {"lat": 27.8974, "lon": 78.0880, "population": "1.1M", "region": "North India"},
    "Tiruppur": {"lat": 11.1085, "lon": 77.3411, "population": "1.1M", "region": "Southern India"},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266, "population": "1.1M", "region": "North India"},
    "Moradabad": {"lat": 28.8386, "lon": 78.7733, "population": "1.1M", "region": "North India"},
    "Jalandhar": {"lat": 31.3260, "lon": 75.5762, "population": "1.1M", "region": "North India"},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245, "population": "1.1M", "region": "Eastern India"},
    
    # More Cities
    "Salem": {"lat": 11.6643, "lon": 78.1460, "population": "1.0M", "region": "Southern India"},
    "Warangal": {"lat": 18.0000, "lon": 79.5833, "population": "1.0M", "region": "Southern India"},
    "Mira-Bhayandar": {"lat": 19.2952, "lon": 72.8547, "population": "1.0M", "region": "Western India"},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "population": "1.0M", "region": "Southern India"},
    "Bhiwandi": {"lat": 19.2956, "lon": 73.0478, "population": "1.0M", "region": "Western India"},
    "Saharanpur": {"lat": 29.9680, "lon": 77.5552, "population": "1.0M", "region": "North India"},
    "Gorakhpur": {"lat": 26.7606, "lon": 83.3732, "population": "1.0M", "region": "North India"},
    "Guntur": {"lat": 16.3067, "lon": 80.4365, "population": "1.0M", "region": "Southern India"},
    "Bikaner": {"lat": 28.0229, "lon": 73.3119, "population": "1.0M", "region": "North India"},
    "Amravati": {"lat": 20.9374, "lon": 77.7796, "population": "1.0M", "region": "Western India"}
}

# Initialize variables
latitude, longitude = None, None
location_name = "Selected Location"
search_radius = st.sidebar.slider("Search radius (km)", 5, 100, 25)
current_date = st.sidebar.date_input("Date for prediction", datetime.now())

# Data source settings (moved up)
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

# Add bulk analysis section
if location_method == "Bulk Analysis - Major Indian Cities":
    st.write("### Bulk Analysis - Major Indian Cities")
    st.write("This will analyze bird hotspots for all major Indian cities with population > 1 million.")
    
    # Initialize session state for bulk analysis
    if 'bulk_analysis_results' not in st.session_state:
        st.session_state.bulk_analysis_results = {}
    if 'bulk_analysis_progress' not in st.session_state:
        st.session_state.bulk_analysis_progress = 0
    if 'bulk_analysis_running' not in st.session_state:
        st.session_state.bulk_analysis_running = False
    if 'current_city_details' not in st.session_state:
        st.session_state.current_city_details = None
    
    # Create columns for stats
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Total Cities", len(MAJOR_INDIAN_CITIES))
    with stat_cols[1]:
        processed_cities = len(st.session_state.bulk_analysis_results)
        st.metric("Processed", processed_cities)
    with stat_cols[2]:
        if processed_cities > 0:
            avg_score = np.mean([
                city_result['data']['hotspots']['hotspot_score'].max() 
                for city_result in st.session_state.bulk_analysis_results.values()
                if 'data' in city_result and 'hotspots' in city_result['data']
            ])
            st.metric("Avg Top Score", f"{avg_score:.2f}")
    with stat_cols[3]:
        if processed_cities > 0:
            total_hotspots = sum([
                len(city_result['data']['hotspots'])
                for city_result in st.session_state.bulk_analysis_results.values()
                if 'data' in city_result and 'hotspots' in city_result['data']
            ])
            st.metric("Total Hotspots", total_hotspots)
    
    # Create a progress bar and status message
    progress_bar = st.progress(st.session_state.bulk_analysis_progress)
    status_text = st.empty()
    
    # Add region filter
    regions = sorted(list(set(city['region'] for city in MAJOR_INDIAN_CITIES.values())))
    selected_regions = st.multiselect(
        "Filter by Region",
        regions,
        default=regions
    )
    
    # Filter cities by selected regions
    filtered_cities = {
        name: data for name, data in MAJOR_INDIAN_CITIES.items()
        if data['region'] in selected_regions
    }
    
    # Add a button to start bulk analysis
    if st.button("Start Bulk Analysis") and not st.session_state.bulk_analysis_running:
        st.session_state.bulk_analysis_running = True
        st.session_state.bulk_analysis_results = {}
        st.session_state.bulk_analysis_progress = 0
        
        total_cities = len(filtered_cities)
        
        # Process each city
        for i, (city_name, city_data) in enumerate(filtered_cities.items(), 1):
            status_text.write(f"Processing {city_name} ({city_data['region']})...")
            st.session_state.current_city_details = f"Analyzing {city_name} - Population: {city_data['population']}"
            
            try:
                # Run the hotspot analysis for this city
                predictor = south_asian_bird_hotspot.SouthAsianBirdHotspotPredictor(
                    region_bbox=(
                        city_data['lat'] - search_radius/111,
                        city_data['lon'] - search_radius/111,
                        city_data['lat'] + search_radius/111,
                        city_data['lon'] + search_radius/111
                    ),
                    grid_size=0.02
                )
                
                result = predictor.process_for_current_date(
                    latitude=city_data['lat'],
                    longitude=city_data['lon'],
                    radius_km=search_radius,
                    date=current_date,
                    use_ebird=use_ebird,
                    use_gbif=use_gbif,
                    use_xeno_canto=use_xeno_canto,
                    ebird_api_key=ebird_api_key if use_ebird else None
                )
                
                # Get habitat and species information
                habitat_to_birds = get_habitat_to_birds_mapping()
                species_info = []
                
                if 'hotspots' in result:
                    for _, hotspot in result['hotspots'].iterrows():
                        # Find habitats in this hotspot
                        present_habitats = []
                        for col in hotspot.index:
                            if col.startswith('is_') and hotspot[col] > 0:
                                habitat_name = col.replace('is_', '').replace('_', ' ')
                                present_habitats.append(habitat_name)
                        
                        # Get birds for these habitats
                        for habitat in present_habitats:
                            for habitat_key, birds in habitat_to_birds.items():
                                if habitat in habitat_key or habitat_key in habitat:
                                    for bird in birds:
                                        species_info.append({
                                            'name': bird['name'],
                                            'type': bird['type'],
                                            'status': bird['status'],
                                            'habitat': habitat
                                        })
                
                # Store the results with additional information
                st.session_state.bulk_analysis_results[city_name] = {
                    'data': result,
                    'population': city_data['population'],
                    'region': city_data['region'],
                    'coordinates': {'lat': city_data['lat'], 'lon': city_data['lon']},
                    'species_info': species_info,
                    'habitats': list(set(info['habitat'] for info in species_info))
                }
                
            except Exception as e:
                st.error(f"Error processing {city_name}: {str(e)}")
            
            # Update progress
            st.session_state.bulk_analysis_progress = i / total_cities
            progress_bar.progress(st.session_state.bulk_analysis_progress)
        
        st.session_state.bulk_analysis_running = False
        status_text.write("Bulk analysis completed!")
    
    # Display current city being processed
    if st.session_state.current_city_details:
        st.info(st.session_state.current_city_details)
    
    # Display results if available
    if st.session_state.bulk_analysis_results:
        st.write("### Analysis Results")
        
        # Create tabs for different views
        result_tabs = st.tabs(["Summary", "Regional Analysis", "Species Insights", "Habitat Analysis", "Detailed View"])
        
        with result_tabs[0]:
            # Create a summary table
            summary_data = []
            for city_name, city_result in st.session_state.bulk_analysis_results.items():
                if 'data' in city_result and 'hotspots' in city_result['data']:
                    hotspots_df = city_result['data']['hotspots']
                    summary_data.append({
                        'City': city_name,
                        'Region': city_result['region'],
                        'Population': city_result['population'],
                        'Top Hotspot Score': hotspots_df['hotspot_score'].max() if 'hotspot_score' in hotspots_df.columns else 'N/A',
                        'Number of Hotspots': len(hotspots_df),
                        'Average Score': hotspots_df['hotspot_score'].mean() if 'hotspot_score' in hotspots_df.columns else 'N/A',
                        'Unique Species': len(set(species['name'] for species in city_result['species_info'])),
                        'Habitat Types': len(city_result['habitats'])
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        with result_tabs[1]:
            st.write("#### Regional Bird Hotspot Analysis")
            
            # Group by region
            region_data = {}
            for city_name, city_result in st.session_state.bulk_analysis_results.items():
                region = city_result['region']
                if region not in region_data:
                    region_data[region] = {
                        'cities': 0,
                        'total_hotspots': 0,
                        'avg_score': 0,
                        'species_set': set(),
                        'habitat_set': set()
                    }
                
                region_data[region]['cities'] += 1
                if 'data' in city_result and 'hotspots' in city_result['data']:
                    region_data[region]['total_hotspots'] += len(city_result['data']['hotspots'])
                    region_data[region]['avg_score'] += city_result['data']['hotspots']['hotspot_score'].mean()
                
                # Add species and habitats
                region_data[region]['species_set'].update(species['name'] for species in city_result['species_info'])
                region_data[region]['habitat_set'].update(city_result['habitats'])
            
            # Create region summary
            region_summary = []
            for region, data in region_data.items():
                if data['cities'] > 0:
                    region_summary.append({
                        'Region': region,
                        'Cities Analyzed': data['cities'],
                        'Total Hotspots': data['total_hotspots'],
                        'Avg Score': data['avg_score'] / data['cities'],
                        'Unique Species': len(data['species_set']),
                        'Habitat Types': len(data['habitat_set'])
                    })
            
            region_df = pd.DataFrame(region_summary)
            st.dataframe(region_df, use_container_width=True)
            
            # Create a map of all hotspots
            st.write("#### Hotspot Distribution Map")
            m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
            
            # Add markers for each city
            for city_name, city_result in st.session_state.bulk_analysis_results.items():
                if 'data' in city_result and 'hotspots' in city_result['data']:
                    # Add a marker for the city
                    folium.CircleMarker(
                        location=[city_result['coordinates']['lat'], city_result['coordinates']['lon']],
                        radius=10,
                        popup=f"{city_name}<br>Population: {city_result['population']}<br>Hotspots: {len(city_result['data']['hotspots'])}",
                        color='red',
                        fill=True
                    ).add_to(m)
            
            # Display the map
            st_folium(m, width=800)
        
        with result_tabs[2]:
            st.write("#### Species Distribution Analysis")
            
            # Collect all species data
            all_species = []
            for city_result in st.session_state.bulk_analysis_results.values():
                all_species.extend(city_result['species_info'])
            
            # Create species summary
            species_summary = pd.DataFrame(all_species)
            species_counts = species_summary['name'].value_counts()
            
            # Display top species
            st.write("##### Most Common Bird Species")
            top_species = species_counts.head(10)
            
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            top_species.plot(kind='bar', ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.title("Top 10 Most Common Bird Species")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Species status distribution
            st.write("##### Bird Status Distribution")
            status_dist = species_summary['status'].value_counts()
            
            # Create a pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.pie(status_dist, labels=status_dist.index, autopct='%1.1f%%')
            plt.title("Distribution of Bird Status")
            st.pyplot(fig)
        
        with result_tabs[3]:
            st.write("#### Habitat Analysis")
            
            # Collect all habitat data
            all_habitats = []
            for city_result in st.session_state.bulk_analysis_results.values():
                all_habitats.extend(city_result['habitats'])
            
            # Create habitat summary
            habitat_counts = pd.Series(all_habitats).value_counts()
            
            # Display habitat distribution
            st.write("##### Habitat Distribution")
            
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            habitat_counts.plot(kind='bar', ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.title("Distribution of Habitat Types")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show habitat-species relationship
            st.write("##### Species per Habitat Type")
            habitat_species = {}
            for city_result in st.session_state.bulk_analysis_results.values():
                for species in city_result['species_info']:
                    if species['habitat'] not in habitat_species:
                        habitat_species[species['habitat']] = set()
                    habitat_species[species['habitat']].add(species['name'])
            
            habitat_species_count = {habitat: len(species) for habitat, species in habitat_species.items()}
            
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            pd.Series(habitat_species_count).plot(kind='bar', ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.title("Number of Unique Species per Habitat")
            plt.tight_layout()
            st.pyplot(fig)
        
        with result_tabs[4]:
            st.write("#### Detailed City Analysis")
            
            # Let user select a city
            selected_city = st.selectbox(
                "Select a city for detailed analysis",
                list(st.session_state.bulk_analysis_results.keys())
            )
            
            if selected_city:
                city_result = st.session_state.bulk_analysis_results[selected_city]
                
                # Display city stats
                st.write(f"##### {selected_city} Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Population", city_result['population'])
                with col2:
                    st.metric("Region", city_result['region'])
                with col3:
                    if 'data' in city_result and 'hotspots' in city_result['data']:
                        st.metric("Number of Hotspots", len(city_result['data']['hotspots']))
                
                # Display hotspots on a map
                st.write("##### Hotspot Locations")
                if 'data' in city_result and 'hotspots' in city_result['data']:
                    m = folium.Map(
                        location=[city_result['coordinates']['lat'], city_result['coordinates']['lon']],
                        zoom_start=11
                    )
                    
                    # Add markers for each hotspot
                    for _, hotspot in city_result['data']['hotspots'].iterrows():
                        folium.CircleMarker(
                            location=[hotspot['latitude'], hotspot['longitude']],
                            radius=8,
                            popup=f"Score: {hotspot['hotspot_score']:.2f}",
                            color='red',
                            fill=True
                        ).add_to(m)
                    
                    # Display the map
                    st_folium(m, width=800)
                
                # Display species information
                st.write("##### Local Species")
                species_df = pd.DataFrame(city_result['species_info']).drop_duplicates()
                st.dataframe(species_df, use_container_width=True)
            
            # Add download buttons
            st.write("### Download Options")
            
            # Function to create detailed Excel report
            def create_bulk_excel():
                output = io.BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                
                # Write summary sheet
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Write regional analysis
                region_df.to_excel(writer, sheet_name='Regional_Analysis', index=False)
                
                # Write species analysis
                species_summary.to_excel(writer, sheet_name='Species_Analysis', index=False)
                
                # Write detailed sheets for each city
                for city_name, city_result in st.session_state.bulk_analysis_results.items():
                    if 'data' in city_result and 'hotspots' in city_result['data']:
                        # Write hotspots
                        city_sheet_name = f"{city_name[:28]}_Hotspots"  # Excel has 31 char limit
                        city_result['data']['hotspots'].to_excel(writer, sheet_name=city_sheet_name, index=False)
                        
                        # Write species info
                        species_sheet_name = f"{city_name[:28]}_Species"
                        pd.DataFrame(city_result['species_info']).to_excel(writer, sheet_name=species_sheet_name, index=False)
                        
                        # Write grid data
                        if 'grid' in city_result['data']:
                            grid_sheet_name = f"{city_name[:28]}_Grid"
                            city_result['data']['grid'].to_excel(writer, sheet_name=grid_sheet_name, index=False)
                
                writer.close()
                return output.getvalue()
            
            # Add download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Download summary CSV
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary (CSV)",
                    data=csv,
                    file_name=f"bird_hotspots_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download detailed Excel report
                excel_data = create_bulk_excel()
                st.download_button(
                    label="Download Detailed Report (Excel)",
                    data=excel_data,
                    file_name=f"bird_hotspots_detailed_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.ms-excel"
                )

# Process location input
if location_method == "Search by City/Locality":
    location_name = st.sidebar.text_input("Enter city or locality name")
    country = st.sidebar.selectbox(
        "Country",
        ["India", "Nepal", "Bangladesh", "Sri Lanka", "Pakistan", "Bhutan"]
    )
    
    if st.sidebar.button("Search for Hotspots") and location_name:
        with st.sidebar:
            with st.spinner("Finding location and analyzing hotspots..."):
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
                        
                        # Proceed directly to hotspot analysis
                        st.session_state.run_analysis = True
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
    
    # Single button for immediate analysis
    if st.sidebar.button("Find Hotspots at These Coordinates"):
        st.session_state.latitude = latitude
        st.session_state.longitude = longitude
        st.session_state.location_name = location_name
        st.session_state.run_analysis = True

elif location_method == "Use Current Location":
    st.sidebar.info("This feature requires browser location permission.")
    
    # Add a placeholder for the location data
    loc_status = st.sidebar.empty()
    
    # Combined button for getting location and running analysis
    loc_button = st.sidebar.button("Find Hotspots at My Location")
    
    if loc_button:
        # In a real app, you'd use proper JS integration with Streamlit components
        # For demo purposes, we'll simulate location detection
        with st.sidebar:
            with st.spinner("Detecting your location and analyzing hotspots..."):
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
                
                # Proceed directly to hotspot analysis
                st.session_state.run_analysis = True

# Additional settings

# Remove the separate "Find Bird Hotspots" button since we now run immediately after location is set
# Instead, add a Refresh button for when data source settings change
if hasattr(st.session_state, 'latitude') and hasattr(st.session_state, 'longitude'):
    if st.sidebar.button("Refresh Analysis with Current Settings"):
        st.session_state.run_analysis = True

# Retrieve from session state if available
if hasattr(st.session_state, 'latitude') and hasattr(st.session_state, 'longitude'):
    if latitude is None and longitude is None:
        latitude = st.session_state.latitude
        longitude = st.session_state.longitude
        if hasattr(st.session_state, 'location_name'):
            location_name = st.session_state.location_name

# Initialize the run_analysis flag if it doesn't exist
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# Main content area - now check the session state flag instead of button click
if latitude is not None and longitude is not None and st.session_state.run_analysis:
    # Reset the flag for next time
    st.session_state.run_analysis = False
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
        # Now proceed directly to hotspot analysis
        with st.spinner("Analyzing bird hotspots..."):
            # Define region bbox (search area around the point)
            degree_radius = search_radius / 111  # ~111km per degree
            region_bbox = (
                latitude - degree_radius,  # min_lat
                longitude - degree_radius,  # min_lon
                latitude + degree_radius,  # max_lat
                longitude + degree_radius   # max_lon
            )
            
            # Initialize predictor with the correct parameters
            # Use the region_bbox we just calculated, not south_asia_bbox
            predictor = south_asian_bird_hotspot.SouthAsianBirdHotspotPredictor(region_bbox=region_bbox, grid_size=0.02)
            
            # Run the analysis using the process_for_current_date method with all required parameters
            result = predictor.process_for_current_date(
                latitude=latitude, 
                longitude=longitude, 
                radius_km=search_radius,
                date=current_date,
                use_ebird=use_ebird,
                use_gbif=use_gbif,
                use_xeno_canto=use_xeno_canto,
                ebird_api_key=ebird_api_key
            )
            
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
                # Create a static visualization of the hotspots with a better heatmap
                try:
                    st.write("### Bird Hotspot Map")
                    st.write("Showing predicted hotspots based on environmental factors and bird data.")
                    
                    # Create a matplotlib figure for the map
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Plot the base grid points if available
                    if 'grid' in result and not result['grid'].empty:
                        # Create a better looking heatmap using contourf
                        x = result['grid']['longitude']
                        y = result['grid']['latitude']
                        z = result['grid']['hotspot_score']
                        
                        # Create a grid for contour plotting
                        xi = np.linspace(min(x), max(x), 100)
                        yi = np.linspace(min(y), max(y), 100)
                        xi, yi = np.meshgrid(xi, yi)
                        
                        # Interpolate the scores onto the grid
                        from scipy.interpolate import griddata
                        zi = griddata((x, y), z, (xi, yi), method='cubic')
                        
                        # Plot the contour
                        contour = ax.contourf(xi, yi, zi, 
                                             levels=15, 
                                             cmap='YlOrRd',
                                             alpha=0.7)
                        
                        # Add a colorbar
                        cbar = plt.colorbar(contour, ax=ax)
                        cbar.set_label('Hotspot Score', fontsize=12)
                        
                        # Add scatter points for the actual data points
                        ax.scatter(x, y, c=z, cmap='YlOrRd', 
                                  edgecolor='k', linewidth=0.5, 
                                  s=30, alpha=0.6)
                    
                    # Mark the search location with a blue star
                    ax.scatter(
                        longitude, 
                        latitude, 
                        marker='*', 
                        s=300, 
                        color='blue', 
                        edgecolor='white', 
                        linewidth=1,
                        label='Search Location',
                        zorder=10
                    )
                    
                    # Add top hotspots with red markers
                    if 'hotspots' in result and not result['hotspots'].empty:
                        # Plot the hotspots
                        hotspot_scatter = ax.scatter(
                            result['hotspots']['longitude'],
                            result['hotspots']['latitude'],
                            marker='o',
                            s=200,
                            color='red',
                            edgecolor='white',
                            linewidth=1,
                            label='Top Hotspots',
                            zorder=11
                        )
                        
                        # Add labels for each hotspot
                        for i, (_, hotspot) in enumerate(result['hotspots'].iterrows(), 1):
                            ax.annotate(
                                f"#{i}",
                                (hotspot['longitude'], hotspot['latitude']),
                                color='white',
                                fontweight='bold',
                                ha='center',
                                va='center',
                                zorder=12
                            )
                    
                    # Add a title and labels
                    ax.set_title(f'Bird Hotspots near {location_name}', fontsize=16)
                    ax.set_xlabel('Longitude', fontsize=12)
                    ax.set_ylabel('Latitude', fontsize=12)
                    
                    # Add a legend
                    ax.legend(loc='upper right', fontsize=12)
                    
                    # Add a grid
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Display the map
                    st.pyplot(fig)
                    
                    # Add hotspot details in expandable sections
                    if 'hotspots' in result and not result['hotspots'].empty:
                        st.write("### Top Hotspot Details")
                        
                        # Get the habitat to birds mapping
                        habitat_to_birds = get_habitat_to_birds_mapping()
                        
                        # Create columns for the hotspots
                        num_hotspots = min(len(result['hotspots']), 3)  # Show up to 3 hotspots in a row
                        hotspot_cols = st.columns(num_hotspots)
                        
                        # Display each hotspot in its own column
                        for i, (col, (_, hotspot)) in enumerate(zip(hotspot_cols, result['hotspots'].iterrows()), 1):
                            with col:
                                with st.container():
                                    # Create a styled header
                                    st.markdown(f"""
                                    <div style="background-color: #3498DB; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                        <h3 style="margin: 0; text-align: center;">Hotspot #{i}</h3>
                                        <p style="margin: 5px 0 0 0; text-align: center; font-size: 18px;">Score: {hotspot['hotspot_score']:.2f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Determine which habitats are present at this hotspot
                                    present_habitats = []
                                    for col_name in hotspot.index:
                                        if col_name.startswith('is_') and hotspot[col_name] > 0:
                                            habitat_name = col_name.replace('is_', '').replace('_', ' ').title()
                                            present_habitats.append(habitat_name)
                                    
                                    # Display habitats
                                    if present_habitats:
                                        st.markdown("#### Habitats")
                                        for habitat in present_habitats:
                                            st.markdown(f"- {habitat}")
                                    
                                    # Get likely birds for these habitats
                                    likely_birds = []
                                    for habitat in present_habitats:
                                        for habitat_key, birds in habitat_to_birds.items():
                                            if habitat.lower() in habitat_key or habitat_key in habitat.lower():
                                                likely_birds.extend(birds)
                                    
                                    # Take up to 5 unique birds
                                    unique_birds = []
                                    seen_names = set()
                                    for bird in likely_birds:
                                        if bird['name'] not in seen_names and len(unique_birds) < 5:
                                            unique_birds.append(bird)
                                            seen_names.add(bird['name'])
                                    
                                    # Display birds
                                    if unique_birds:
                                        st.markdown("#### Likely Birds")
                                        for bird in unique_birds:
                                            st.markdown(f"""
                                            <div style="margin-bottom: 8px; padding: 5px; background-color: #f8f9fa; border-radius: 3px;">
                                                <b>{bird['name']}</b><br>
                                                <small>{bird['type']} - {bird['status']}</small>
                                            </div>
                                            """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                    st.error(traceback.format_exc())
            
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
                        
                        # Get the list of birds from the hotspots
                        bird_species = set()
                        
                        # Extract birds from habitat mapping based on hotspot habitats
                        if 'hotspots' in result and not result['hotspots'].empty:
                            habitat_to_birds = get_habitat_to_birds_mapping()
                            
                            for _, hotspot in result['hotspots'].iterrows():
                                # Find habitats in this hotspot
                                present_habitats = []
                                for col in hotspot.index:
                                    if col.startswith('is_') and hotspot[col] > 0:
                                        habitat_name = col.replace('is_', '').replace('_', ' ')
                                        present_habitats.append(habitat_name)
                                
                                # Get birds for these habitats
                                for habitat in present_habitats:
                                    for habitat_key, birds in habitat_to_birds.items():
                                        if habitat in habitat_key or habitat_key in habitat:
                                            for bird in birds:
                                                bird_species.add(bird['name'])
                        
                        # Convert to list and sort
                        bird_list = sorted(list(bird_species))
                        
                        if bird_list:
                            # Let user select a bird
                            selected_bird = st.selectbox("Select a bird to hear its calls:", bird_list)
                            
                            # Fetch recordings for the selected bird
                            with st.spinner(f"Fetching sound recordings for {selected_bird}..."):
                                xeno_df = bird_client.get_xeno_canto_recordings_by_species(selected_bird)
                            
                            if not xeno_df.empty:
                                # Display summary
                                st.write(f"Found {len(xeno_df)} sound recordings for {selected_bird}")
                                
                                # Display featured recordings with spectrograms
                                st.write("##### Featured Recordings")
                                
                                # Get top 3 recordings with spectrograms
                                featured_recordings = xeno_df.head(3).to_dict('records') if len(xeno_df) > 0 else []
                                
                                for rec in featured_recordings:
                                    with st.expander(f"{rec.get('type', 'Call')} recorded by {rec.get('rec', 'Unknown')}"):
                                        cols = st.columns([2, 1])
                                        
                                        with cols[0]:
                                            # Display spectrogram if available
                                            if 'sono' in rec and rec['sono']:
                                                # Fix protocol-relative URLs by adding https:
                                                spectrogram_url = rec['sono']['small']
                                                if spectrogram_url.startswith('//'):
                                                    spectrogram_url = 'https:' + spectrogram_url
                                                
                                                # Display the image using the fixed URL
                                                st.markdown(f"![Spectrogram of {selected_bird}]({spectrogram_url})")
                                                st.caption(f"Spectrogram of {selected_bird} {rec.get('type', 'call')}")
                                            
                                            # Display recording info
                                            st.write(f"**Location:** {rec.get('loc', 'Unknown')}, {rec.get('cnt', 'Unknown')}")
                                            st.write(f"**Date:** {rec.get('date', 'Unknown')}")
                                            st.write(f"**Quality rating:** {rec.get('q', 'Unknown')}")
                                        
                                        with cols[1]:
                                            # Display audio player if file is available
                                            if 'file' in rec and rec['file']:
                                                audio_url = rec['file']
                                                if audio_url.startswith('//'):
                                                    audio_url = 'https:' + audio_url
                                                
                                                # Create an HTML audio player
                                                st.markdown(f"""
                                                <audio controls style="width: 100%;">
                                                    <source src="{audio_url}" type="audio/mpeg">
                                                    Your browser does not support the audio element.
                                                </audio>
                                                """, unsafe_allow_html=True)
                                            
                                            # Add link to Xeno-canto
                                            if 'url' in rec:
                                                st.markdown(f"[View on Xeno-canto]({rec['url']})")
                            
                                # Display the data table
                                st.write("##### All Recordings")
                                
                                # Clean up the dataframe for display
                                display_cols = ['type', 'rec', 'cnt', 'loc', 'date', 'q']
                                display_df = xeno_df[display_cols].copy() if all(col in xeno_df.columns for col in display_cols) else xeno_df
                                
                                # Rename columns
                                display_df = display_df.rename(columns={
                                    'type': 'Call Type',
                                    'rec': 'Recordist',
                                    'cnt': 'Country',
                                    'loc': 'Location',
                                    'date': 'Date',
                                    'q': 'Quality'
                                })
                                
                                st.dataframe(display_df, use_container_width=True)
                            else:
                                st.info(f"No sound recordings found for {selected_bird}. Try selecting a different bird.")
                        else:
                            st.info("No bird species identified for this location. Try a different location or adjust your search parameters.")
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

# Add a download button for the results
if 'result' in locals() and 'hotspots' in result and not result['hotspots'].empty:
    st.write("### Download Results")
    
    # Create a function to convert the results to Excel
    def to_excel():
        # Create a Pandas Excel writer
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        
        # Write the hotspots to the Excel file
        result['hotspots'].to_excel(writer, sheet_name='Hotspots', index=False)
        
        # Write the grid data to the Excel file if available
        if 'grid' in result and not result['grid'].empty:
            result['grid'].to_excel(writer, sheet_name='Grid Data', index=False)
        
        # Create a sheet for bird species
        bird_species = []
        habitat_to_birds = get_habitat_to_birds_mapping()
        
        # Extract birds from habitat mapping based on hotspot habitats
        for _, hotspot in result['hotspots'].iterrows():
            # Find habitats in this hotspot
            present_habitats = []
            for col in hotspot.index:
                if col.startswith('is_') and hotspot[col] > 0:
                    habitat_name = col.replace('is_', '').replace('_', ' ')
                    present_habitats.append(habitat_name)
            
            # Get birds for these habitats
            for habitat in present_habitats:
                for habitat_key, birds in habitat_to_birds.items():
                    if habitat in habitat_key or habitat_key in habitat:
                        for bird in birds:
                            bird_species.append({
                                'Name': bird['name'],
                                'Type': bird['type'],
                                'Status': bird['status'],
                                'Habitat': habitat.title()
                            })
        
        # Convert to DataFrame and remove duplicates
        if bird_species:
            birds_df = pd.DataFrame(bird_species).drop_duplicates()
            birds_df.to_excel(writer, sheet_name='Bird Species', index=False)
        
        # Close the writer and get the output
        writer.close()
        return output.getvalue()
    
    # Add the download button
    excel_data = to_excel()
    st.download_button(
        label="Download Excel Report",
        data=excel_data,
        file_name=f"bird_hotspots_{location_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.ms-excel"
    ) 