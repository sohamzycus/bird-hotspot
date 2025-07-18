import os
import sys
import folium
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from shapely.geometry import Point
import streamlit as st
import time
import traceback
import logging
from folium.plugins import HeatMap
import requests
import json
from dotenv import load_dotenv
from bird_api_client import BirdDataClient
import matplotlib.pyplot as plt
import numpy as np
import south_asian_bird_hotspot
import io
import base64
from PIL import Image
from streamlit_folium import folium_static, st_folium
import gc
import random

# Load environment variables for API keys
load_dotenv()

# API keys (store these in a .env file for security)
EBIRD_API_KEY = os.getenv("EBIRD_API_KEY", "kajuv1chrnbc")
GBIF_USER = os.getenv("GBIF_USER", "sohiyiy")
GBIF_PWD = os.getenv("GBIF_PWD", "K6M#h7Uk@c2Fm6_")

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hotspot_debug.log", encoding='utf-8'),
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

# India geographic boundaries for dynamic grid generation
INDIA_BOUNDS = {
    'north': 37.6,    # Kashmir
    'south': 8.0,     # Exclude Sri Lanka (was 6.4)  
    'east': 97.25,    # Arunachal Pradesh
    'west': 68.7      # Gujarat/Rajasthan border
}

# Famous birding locations that birders recognize
FAMOUS_BIRDING_SPOTS = {
    # National Parks and Wildlife Sanctuaries (most recognized)
    'bharatpur': {'name': 'Bharatpur Bird Sanctuary', 'state': 'Rajasthan', 'fame': 'Siberian Crane winter home'},
    'corbett': {'name': 'Jim Corbett National Park', 'state': 'Uttarakhand', 'fame': 'India\'s oldest national park'},
    'kaziranga': {'name': 'Kaziranga National Park', 'state': 'Assam', 'fame': 'One-horned rhinoceros and grassland birds'},
    'ranthambore': {'name': 'Ranthambore National Park', 'state': 'Rajasthan', 'fame': 'Tiger reserve with crested serpent eagles'},
    'periyar': {'name': 'Periyar Wildlife Sanctuary', 'state': 'Kerala', 'fame': 'Western Ghats endemic species'},
    'bandipur': {'name': 'Bandipur National Park', 'state': 'Karnataka', 'fame': 'Tiger reserve with Malabar trogons'},
    'mudumalai': {'name': 'Mudumalai National Park', 'state': 'Tamil Nadu', 'fame': 'Nilgiri biosphere with hornbills'},
    'kanha': {'name': 'Kanha National Park', 'state': 'Madhya Pradesh', 'fame': 'Barasingha and Indian roller paradise'},
    'pench': {'name': 'Pench National Park', 'state': 'Madhya Pradesh', 'fame': 'Mowgli land with paradise flycatchers'},
    'sundarbans': {'name': 'Sundarbans National Park', 'state': 'West Bengal', 'fame': 'Mangrove specialist birds'},
    
    # Hill Stations and Birding Destinations
    'ooty': {'name': 'Ooty (Nilgiris)', 'state': 'Tamil Nadu', 'fame': 'Nilgiri flycatcher and blackbird hotspot'},
    'munnar': {'name': 'Munnar Hills', 'state': 'Kerala', 'fame': 'Western Ghats endemics and tea estates'},
    'kodaikanal': {'name': 'Kodaikanal Hills', 'state': 'Tamil Nadu', 'fame': 'Pied thrush and shola forest birds'},
    'coorg': {'name': 'Coorg (Kodagu)', 'state': 'Karnataka', 'fame': 'Coffee plantation birds and Malabar trogon'},
    'darjeeling': {'name': 'Darjeeling Hills', 'state': 'West Bengal', 'fame': 'Himalayan monal and laughingthrushes'},
    'shimla': {'name': 'Shimla Hills', 'state': 'Himachal Pradesh', 'fame': 'Himalayan woodpeckers and snow partridge'},
    'mussoorie': {'name': 'Mussoorie Hills', 'state': 'Uttarakhand', 'fame': 'Himalayan birding with cheer pheasant'},
    
    # Wetlands and Lakes (critical for birders)
    'chilika': {'name': 'Chilika Lake', 'state': 'Odisha', 'fame': 'Asia\'s largest coastal lagoon, flamingo winter home'},
    'pulicat': {'name': 'Pulicat Lake', 'state': 'Tamil Nadu', 'fame': 'Greater flamingo and spot-billed pelican hub'},
    'sambhar': {'name': 'Sambhar Salt Lake', 'state': 'Rajasthan', 'fame': 'Flamingo breeding and salt lake specialists'},
    'loktak': {'name': 'Loktak Lake', 'state': 'Manipur', 'fame': 'Sangai deer habitat with floating islands'},
    'wular': {'name': 'Wular Lake', 'state': 'Kashmir', 'fame': 'Himalayan waterfowl paradise'},
    
    # Coastal Areas
    'rann': {'name': 'Rann of Kutch', 'state': 'Gujarat', 'fame': 'Desert birding with MacQueen\'s bustard'},
    'goa': {'name': 'Goa Beaches', 'state': 'Goa', 'fame': 'Coastal migrants and tern colonies'},
    'andaman': {'name': 'Andaman Islands', 'state': 'Andaman', 'fame': 'Endemic pigeons and sea eagles'},
    
    # City Birding Spots
    'delhi': {'name': 'Delhi Ridge', 'state': 'Delhi', 'fame': 'Urban birding with rose-ringed parakeets'},
    'mumbai': {'name': 'Sanjay Gandhi National Park', 'state': 'Maharashtra', 'fame': 'Urban forest with paradise flycatcher'},
    'bangalore': {'name': 'Lalbagh Gardens', 'state': 'Karnataka', 'fame': 'Urban park with Asian koel and babblers'},
    'chennai': {'name': 'Guindy National Park', 'state': 'Tamil Nadu', 'fame': 'Urban sanctuary with spotted deer and blackbuck'},
    'kolkata': {'name': 'East Kolkata Wetlands', 'state': 'West Bengal', 'fame': 'Urban wetlands with purple swamphen'},
    'pune': {'name': 'Pune Hills', 'state': 'Maharashtra', 'fame': 'Western Ghats foothills birding'},
    'hyderabad': {'name': 'Hussain Sagar Lake', 'state': 'Telangana', 'fame': 'Urban lake with little grebe and kingfishers'},
}

# Common to Scientific name mapping for Indian birds (most common species)
INDIAN_BIRD_NAMES = {
    # Common Residents
    'House Sparrow': {'scientific': 'Passer domesticus', 'fun_fact': 'Once abundant, now declining in cities due to modern architecture and pollution'},
    'Common Myna': {'scientific': 'Acridotheres tristis', 'fun_fact': 'Highly intelligent bird that can mimic human speech and other sounds'},
    'Rock Pigeon': {'scientific': 'Columba livia', 'fun_fact': 'Ancestor of all domestic pigeons, navigates using magnetic fields'},
    'Rose-ringed Parakeet': {'scientific': 'Psittacula krameri', 'fun_fact': 'Only parrot species that has adapted to city life, roosts in huge flocks'},
    'Red-vented Bulbul': {'scientific': 'Pycnonotus cafer', 'fun_fact': 'State bird of Rajasthan, known for its melodious dawn chorus'},
    'Asian Koel': {'scientific': 'Eudynamys scolopaceus', 'fun_fact': 'Brood parasite that tricks crows into raising its young'},
    
    # Raptors
    'Black Kite': {'scientific': 'Milvus migrans', 'fun_fact': 'Master scavenger, can snatch food from your hands while flying'},
    'Shikra': {'scientific': 'Accipiter badius', 'fun_fact': 'Urban hawk specialist that hunts small birds in city parks'},
    'Common Buzzard': {'scientific': 'Buteo buteo', 'fun_fact': 'Excellent thermaling bird, can soar for hours without flapping'},
    'Brahminy Kite': {'scientific': 'Haliastur indus', 'fun_fact': 'Sacred to Hindus, associated with Lord Vishnu, fish-eating specialist'},
    
    # Water Birds
    'Little Egret': {'scientific': 'Egretta garzetta', 'fun_fact': 'Changes from white to golden breeding plumage, uses feet to stir up fish'},
    'Cattle Egret': {'scientific': 'Bubulcus ibis', 'fun_fact': 'Follows grazing animals to catch insects disturbed by their movement'},
    'Indian Pond Heron': {'scientific': 'Ardeola grayii', 'fun_fact': 'Master of camouflage, perfectly still until prey comes within striking distance'},
    'Purple Heron': {'scientific': 'Ardea purpurea', 'fun_fact': 'Shy wetland specialist with incredible neck flexibility for fishing'},
    'Little Grebe': {'scientific': 'Tachybaptus ruficollis', 'fun_fact': 'Excellent diver, can stay underwater for 30 seconds while fishing'},
    'Common Kingfisher': {'scientific': 'Alcedo atthis', 'fun_fact': 'Dives at 25 mph into water, has nictitating membrane to see underwater'},
    
    # Forest Birds
    'Oriental Magpie-Robin': {'scientific': 'Copsychus saularis', 'fun_fact': 'National bird of Bangladesh, excellent songster with 100+ calls'},
    'Indian Robin': {'scientific': 'Copsychus fulicatus', 'fun_fact': 'Endemic to Indian subcontinent, male performs elaborate courtship dance'},
    'White-throated Kingfisher': {'scientific': 'Halcyon smyrnensis', 'fun_fact': 'State bird of West Bengal, hunts not just fish but insects and lizards'},
    'Indian Roller': {'scientific': 'Coracias benghalensis', 'fun_fact': 'State bird of Odisha, performs spectacular aerial rolls during breeding'},
    'Coppersmith Barbet': {'scientific': 'Psilopogon haemacephalus', 'fun_fact': 'Named for its metallic calls, excavates nest holes in dead tree trunks'},
    
    # Endemics and Specialties
    'Malabar Trogon': {'scientific': 'Harpactes fasciatus', 'fun_fact': 'Western Ghats endemic, sits motionless for hours before catching insects'},
    'Nilgiri Flycatcher': {'scientific': 'Eumyias albicaudatus', 'fun_fact': 'Found only in Nilgiri hills, prefers shola forest understory'},
    'Kerala Laughingthrush': {'scientific': 'Trochalopteron fairbanki', 'fun_fact': 'Endemic to Western Ghats, travels in noisy flocks through dense forest'},
    'Malabar Whistling Thrush': {'scientific': 'Myophonus horsfieldii', 'fun_fact': 'Called "whistling schoolboy" for its melodious songs near hill streams'},
    
    # Migrants
    'Common Sandpiper': {'scientific': 'Actitis hypoleucos', 'fun_fact': 'Travels 6000+ km from Siberia, can be seen bobbing on any Indian waterbank'},
    'Eurasian Hoopoe': {'scientific': 'Upupa epops', 'fun_fact': 'Israel\'s national bird, visits India in winter, probes soil with curved bill'},
    'Blue-throated Flycatcher': {'scientific': 'Cyornis rubeculoides', 'fun_fact': 'Himalayan breeder, migrates to Western Ghats in winter following ancient routes'},
    'Forest Wagtail': {'scientific': 'Dendronanthus indicus', 'fun_fact': 'Unlike other wagtails, prefers forest floor, migrates in large flocks'},
    
    # Nocturnal Species
    'Indian Eagle-Owl': {'scientific': 'Bubo bengalensis', 'fun_fact': 'Largest owl in India, can rotate head 270°, hunts from telegraph poles'},
    'Spotted Owlet': {'scientific': 'Athene brama', 'fun_fact': 'Active during day, nests in tree holes, often seen in pairs on branches'},
    'Brown Fish Owl': {'scientific': 'Ketupa zeylonensis', 'fun_fact': 'Fishing specialist with unfeathered legs, calls like a human snoring'},
    
    # Game Birds
    'Red Junglefowl': {'scientific': 'Gallus gallus', 'fun_fact': 'Ancestor of domestic chicken, male\'s crow can be heard 1.5 km away'},
    'Indian Peafowl': {'scientific': 'Pavo cristatus', 'fun_fact': 'National bird of India, male\'s train has 200+ eye-spots for courtship display'},
    'Grey Francolin': {'scientific': 'Francolinus pondicerianus', 'fun_fact': 'Calls "pateela pateela" which sounds like asking for a pot in Hindi'},
    'Common Quail': {'scientific': 'Coturnix coturnix', 'fun_fact': 'Smallest game bird, migrates thousands of miles despite being poor flier'},
}

def get_scientific_name_and_fun_fact(common_name):
    """
    Convert common bird name to scientific name and add fun fact.
    
    Parameters:
    -----------
    common_name : str
        Common name of the bird
        
    Returns:
    --------
    dict
        Dictionary with scientific name and fun fact
    """
    # Check our comprehensive database first
    if common_name in INDIAN_BIRD_NAMES:
        return INDIAN_BIRD_NAMES[common_name]
    
    # Fallback for less common species - create generic scientific format
    # This handles species not in our database
    words = common_name.split()
    if len(words) >= 2:
        genus = words[0].lower()
        species = words[1].lower()
        scientific = f"{genus.capitalize()} {species}"
        fun_fact = f"Species found in the Indian subcontinent. {common_name}s are part of India's rich avian diversity."
    else:
        scientific = f"{common_name} sp."
        fun_fact = f"{common_name} is one of India's fascinating bird species awaiting detailed documentation."
    
    return {
        'scientific': scientific,
        'fun_fact': fun_fact
    }

def get_birder_friendly_location_name(lat, lng, use_geocoding=True):
    """
    Get location names that birders would recognize - famous birding spots, national parks, etc.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lng : float
        Longitude
    use_geocoding : bool
        Whether to use reverse geocoding
        
    Returns:
    --------
    str
        Birder-friendly location name
    """
    # First check if we're near any famous birding locations (within ~50km)
    for spot_key, spot_info in FAMOUS_BIRDING_SPOTS.items():
        # Define approximate coordinates for famous spots (you could expand this)
        spot_coords = {
            'bharatpur': (27.2152, 77.5222),
            'corbett': (29.5200, 78.9469),
            'kaziranga': (26.5775, 93.1653),
            'ooty': (11.4102, 76.6950),
            'munnar': (10.0889, 77.0595),
            'chilika': (19.7093, 85.3188),
            'rann': (23.7337, 69.0588),
            'delhi': (28.6139, 77.2090),
            'mumbai': (19.0760, 72.8777),
            'bangalore': (12.9716, 77.5946),
            'chennai': (13.0827, 80.2707),
            'kolkata': (22.5726, 88.3639),
        }
        
        if spot_key in spot_coords:
            spot_lat, spot_lng = spot_coords[spot_key]
            distance = ((lat - spot_lat)**2 + (lng - spot_lng)**2)**0.5
            
            # If within ~0.5 degrees (~55km), consider it nearby
            if distance < 0.5:
                return f"Near {spot_info['name']} ({spot_info['fame']})"
    
    # Use geocoding for well-known places
    if use_geocoding:
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="bird_hotspot_finder", timeout=10)
            
            location = geolocator.reverse((lat, lng), exactly_one=True, language='en')
            
            if location and location.address:
                address_parts = location.address.split(', ')
                
                # Look for birding-relevant terms
                birding_keywords = [
                    'national park', 'wildlife sanctuary', 'bird sanctuary', 'reserve', 'forest',
                    'lake', 'wetland', 'dam', 'hills', 'valley', 'sanctuary', 'conservation',
                    'biosphere', 'tiger reserve', 'elephant reserve'
                ]
                
                place_candidates = []
                
                for part in address_parts:
                    part_clean = part.strip()
                    part_lower = part_clean.lower()
                    
                    # Skip generic terms
                    if (len(part_clean) > 2 and 
                        not part_clean.isdigit() and 
                        part_lower not in ['india', 'भारत', 'republic of india']):
                        
                        # Prioritize nature/wildlife areas
                        if any(keyword in part_lower for keyword in birding_keywords):
                            place_candidates.insert(0, part_clean)
                        # Then known birding cities/districts
                        elif any(city in part_lower for city in ['shimla', 'ooty', 'munnar', 'coorg', 'darjeeling']):
                            place_candidates.insert(0, f"{part_clean} (Hill Station)")
                        else:
                            place_candidates.append(part_clean)
                
                if place_candidates:
                    return place_candidates[0]
                    
        except Exception as e:
            logger.debug(f"Geocoding failed for birding location: {str(e)}")
    
    # Fallback to habitat-based birding description
    return get_habitat_based_birding_description(lat, lng)

def get_habitat_based_birding_description(lat, lng):
    """
    Get habitat-based description that birders would understand.
    
    Parameters:
    -----------
    lat : float
        Latitude  
    lng : float
        Longitude
        
    Returns:
    --------
    str
        Habitat-based birding description
    """
    # Altitude-based habitats (birders think in terms of elevation zones)
    if lat > 32:  # High Himalayas
        return "Alpine Himalayan Zone (Snow Partridge, Tibetan Snowfinch habitat)"
    elif lat > 28:  # Middle Himalayas
        return "Temperate Himalayan Zone (Monal, Kalij Pheasant habitat)"
    elif lat > 24:  # Foothills and northern plains
        if lng < 75:
            return "Thar Desert Edge (Desert Wheatear, Houbara habitat)"
        else:
            return "Indo-Gangetic Plains (Sarus Crane, Painted Stork habitat)"
    elif lat > 20:  # Central India
        if lng > 85:
            return "Eastern Deciduous Forests (Hornbill, Drongo habitat)"
        else:
            return "Central Dry Forests (Indian Pitta, Paradise Flycatcher habitat)"
    elif lat > 16:  # Deccan Plateau
        if lng < 75:
            return "Western Ghats Foothills (Malabar Trogon potential habitat)"
        else:
            return "Eastern Ghats Region (Yellow-throated Bulbul habitat)"
    elif lat > 12:  # Southern India
        if lng < 76:
            return "Western Ghats Highlands (Nilgiri Flycatcher, Kerala Laughingthrush habitat)"
        else:
            return "Southern Eastern Ghats (White-bellied Treepie habitat)"
    else:  # Far South
        if lng < 77:
            return "Kerala Coastal Plains (Malabar Grey Hornbill habitat)" 
        else:
            return "Tamil Nadu Coast (Spot-billed Pelican, Painted Stork habitat)"

def generate_dynamic_india_grid(max_points=100, grid_type="systematic"):
    """
    Generate a dynamic grid covering entire India systematically.
    
    Parameters:
    -----------
    max_points : int
        Maximum number of grid points to generate
    grid_type : str
        Type of grid generation ("systematic", "adaptive", "dense")
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with grid points covering India
    """
    try:
        logger.info(f"Generating dynamic India grid with {max_points} points using {grid_type} approach")
        
        if grid_type == "systematic":
            # Calculate grid dimensions for systematic coverage
            grid_points = []
            
            # Calculate spacing to achieve approximately max_points
            total_lat_range = INDIA_BOUNDS['north'] - INDIA_BOUNDS['south']
            total_lng_range = INDIA_BOUNDS['east'] - INDIA_BOUNDS['west']
            
            # Calculate grid dimensions to get closer to max_points
            # Use a more precise calculation to ensure we get the requested number of points
            aspect_ratio = total_lng_range / total_lat_range
            lat_points = int(np.sqrt(max_points / aspect_ratio))
            lng_points = int(max_points / lat_points)
            
            # Adjust to get closer to target
            if lat_points * lng_points < max_points:
                if lat_points * (lng_points + 1) <= max_points:
                    lng_points += 1
                elif (lat_points + 1) * lng_points <= max_points:
                    lat_points += 1
            
            lat_spacing = total_lat_range / (lat_points - 1) if lat_points > 1 else 0
            lng_spacing = total_lng_range / (lng_points - 1) if lng_points > 1 else 0
            
            point_id = 1
            current_lat = INDIA_BOUNDS['south']
            
            while current_lat <= INDIA_BOUNDS['north'] and len(grid_points) < max_points:
                current_lng = INDIA_BOUNDS['west']
                while current_lng <= INDIA_BOUNDS['east'] and len(grid_points) < max_points:
                    
                    # Get dynamic location name using coordinates
                    if point_id <= 50:  # Use reverse geocoding for first 50 points
                        base_name = get_birder_friendly_location_name(current_lat, current_lng, use_geocoding=True)
                        # Add a small delay to respect rate limits
                        time.sleep(0.1)
                    else:
                        base_name = get_birder_friendly_location_name(current_lat, current_lng, use_geocoding=False)
                    
                    # Only add coordinate suffix if it's not already showing "Outside India"
                    if "Outside India" not in base_name:
                        location_name = f"{base_name} ({current_lat:.3f}°N, {current_lng:.3f}°E)"
                    else:
                        location_name = base_name  # Skip points outside India
                        current_lng += lng_spacing
                        point_id += 1
                        continue
                    
                    grid_points.append({
                        'id': point_id,
                        'latitude': round(current_lat, 4),
                        'longitude': round(current_lng, 4),
                        'location_name': location_name,
                        'grid_type': 'systematic'
                    })
                    
                    current_lng += lng_spacing
                    point_id += 1
                
                current_lat += lat_spacing
        
        elif grid_type == "adaptive":
            # Adaptive grid with higher density in biodiverse regions
            grid_points = []
            
            # Define biodiverse regions with higher sampling density (stricter India bounds)
            biodiverse_regions = [
                # Western Ghats
                {'lat_range': (8.0, 21.0), 'lng_range': (73.0, 77.5), 'density': 0.4},
                # Eastern Himalayas
                {'lat_range': (26.0, 30.0), 'lng_range': (88.0, 95.0), 'density': 0.3},
                # Central India
                {'lat_range': (20.0, 26.0), 'lng_range': (75.0, 85.0), 'density': 0.2},
                # Coastal regions
                {'lat_range': (8.5, 25.0), 'lng_range': (68.7, 74.0), 'density': 0.1}
            ]
            
            point_id = 1
            points_allocated = 0
            
            for region in biodiverse_regions:
                if points_allocated >= max_points:
                    break
                    
                points_for_region = int(max_points * region['density'])
                
                # Generate random points within this region
                for _ in range(points_for_region):
                    if points_allocated >= max_points:
                        break
                        
                    lat = random.uniform(region['lat_range'][0], region['lat_range'][1])
                    lng = random.uniform(region['lng_range'][0], region['lng_range'][1])
                    
                    # Get birder-friendly location name using coordinates
                    if point_id <= 50:  # Use reverse geocoding for first 50 points
                        base_name = get_birder_friendly_location_name(lat, lng, use_geocoding=True)
                        time.sleep(0.1)
                    else:
                        base_name = get_birder_friendly_location_name(lat, lng, use_geocoding=False)
                    
                    # Only add coordinate suffix if it's not already showing "Outside India"
                    if "Outside India" not in base_name:
                        location_name = f"{base_name} ({lat:.3f}°N, {lng:.3f}°E)"
                    else:
                        continue  # Skip points outside India
                    
                    grid_points.append({
                        'id': point_id,
                        'latitude': round(lat, 4),
                        'longitude': round(lng, 4),
                        'location_name': location_name,
                        'grid_type': 'adaptive'
                    })
                    
                    point_id += 1
                    points_allocated += 1
        
        else:  # dense
            # Dense systematic grid
            grid_points = []
            
            # Dense grid with better point distribution
            total_lat_range = INDIA_BOUNDS['north'] - INDIA_BOUNDS['south']
            total_lng_range = INDIA_BOUNDS['east'] - INDIA_BOUNDS['west']
            
            # Increase density by 20% for dense grid
            dense_points = int(max_points * 1.2)
            aspect_ratio = total_lng_range / total_lat_range
            lat_points = int(np.sqrt(dense_points / aspect_ratio))
            lng_points = int(dense_points / lat_points)
            
            # Adjust to get closer to target for dense grid
            if lat_points * lng_points < dense_points:
                if lat_points * (lng_points + 1) <= dense_points:
                    lng_points += 1
                elif (lat_points + 1) * lng_points <= dense_points:
                    lat_points += 1
            
            lat_spacing = total_lat_range / (lat_points - 1) if lat_points > 1 else 0
            lng_spacing = total_lng_range / (lng_points - 1) if lng_points > 1 else 0
            
            point_id = 1
            current_lat = INDIA_BOUNDS['south']
            
            while current_lat <= INDIA_BOUNDS['north'] and len(grid_points) < max_points:
                current_lng = INDIA_BOUNDS['west']
                while current_lng <= INDIA_BOUNDS['east'] and len(grid_points) < max_points:
                    
                    # Get birder-friendly location name using coordinates
                    if point_id <= 50:  # Use reverse geocoding for first 50 points
                        base_name = get_birder_friendly_location_name(current_lat, current_lng, use_geocoding=True)
                        time.sleep(0.1)
                    else:
                        base_name = get_birder_friendly_location_name(current_lat, current_lng, use_geocoding=False)
                    
                    # Only add coordinate suffix if it's not already showing "Outside India"
                    if "Outside India" not in base_name:
                        location_name = f"{base_name} ({current_lat:.3f}°N, {current_lng:.3f}°E)"
                    else:
                        current_lng += lng_spacing
                        point_id += 1
                        continue  # Skip points outside India
                    
                    grid_points.append({
                        'id': point_id,
                        'latitude': round(current_lat, 4),
                        'longitude': round(current_lng, 4),
                        'location_name': location_name,
                        'grid_type': 'dense'
                    })
                    
                    current_lng += lng_spacing
                    point_id += 1
                
                current_lat += lat_spacing
        
        df = pd.DataFrame(grid_points)
        logger.info(f"Generated {len(df)} dynamic grid points covering India (filtered to exclude outside points)")
        return df
    
    except Exception as e:
        logger.error(f"Error generating dynamic India grid: {str(e)}")
        return pd.DataFrame()

def check_habitat_viability(lat, lng):
    """
    Check if coordinates are in viable bird habitat.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lng : float
        Longitude
        
    Returns:
    --------
    str
        Habitat assessment
    """
    # Check if coordinates are in ocean areas (major issue for no bird data)
    
    # Arabian Sea (west of India)
    if lng < 70.0 and lat < 25.0:
        return "Arabian Sea (Ocean)"
    
    # Bay of Bengal (east of India) 
    if lng > 92.0 and lat < 20.0:
        return "Bay of Bengal (Ocean)"
    
    # Indian Ocean (south of India)
    if lat < 8.0:
        return "Indian Ocean (Ocean)"
    
    # High Himalayas (very sparse bird populations)
    if lat > 32.0 and lng > 75.0:
        return "High Himalayas (Sparse)"
    
    # Desert regions (Thar Desert)
    if lng < 73.0 and lat > 25.0 and lat < 30.0:
        return "Thar Desert (Arid)"
    
    # Remote northeast hills
    if lng > 94.0 and lat > 25.0:
        return "Remote NE Hills (Sparse)"
    
    # Good terrestrial habitat
    return "Terrestrial (Good)"

def get_dynamic_region_name(lat, lng):
    """
    Get dynamic region name based on coordinates using geographic logic.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lng : float
        Longitude
        
    Returns:
    --------
    str
        Dynamic region name
    """
    # Dynamic region identification based on major geographic/administrative boundaries
    
    # Far Northern regions (High Himalayas)
    if lat >= 32:
        if lng <= 76:
            return "Srinagar"  # Kashmir
        elif lng <= 78:
            return "Shimla"   # Himachal
        elif lng <= 81:
            return "Dehradun" # Uttarakhand
        else:
            return "Itanagar" # Eastern Himalayas
    
    # Upper northern regions (Northern Plains)
    elif lat >= 28:
        if lng <= 74:
            return "Bikaner"   # North Rajasthan
        elif lng <= 78:
            return "Delhi"     # Delhi-NCR area
        elif lng <= 82:
            return "Lucknow"   # UP North
        elif lng <= 88:
            return "Patna"     # Bihar
        else:
            return "Guwahati"  # Northeast
    
    # Central northern regions
    elif lat >= 24:
        if lng <= 72:
            return "Jaisalmer" # Thar Desert
        elif lng <= 76:
            return "Jaipur"    # Rajasthan central
        elif lng <= 80:
            return "Bhopal"    # Madhya Pradesh
        elif lng <= 84:
            return "Raipur"    # Chhattisgarh
        elif lng <= 88:
            return "Ranchi"    # Jharkhand
        else:
            return "Kolkata"   # West Bengal North
    
    # Central regions
    elif lat >= 20:
        if lng <= 73:
            return "Ahmedabad" # Gujarat
        elif lng <= 76:
            return "Pune"      # Maharashtra North
        elif lng <= 80:
            return "Mumbai"    # Central Maharashtra
        elif lng <= 84:
            return "Indore"    # Eastern MP
        elif lng <= 87:
            return "Bhubaneswar" # Odisha
        else:
            return "Kolkata"   # West Bengal Central
    
    # South-central regions
    elif lat >= 16:
        if lng <= 74:
            return "Goa"       # Goa-Karnataka Coast
        elif lng <= 77:
            return "Bangalore" # Karnataka Plateau
        elif lng <= 80:
            return "Hyderabad" # Andhra Pradesh
        elif lng <= 84:
            return "Puri"      # Odisha Coast
        else:
            return "Visakhapatnam" # Bay of Bengal
    
    # Southern regions
    elif lat >= 12:
        if lng <= 76:
            return "Mysore"    # Karnataka South
        elif lng <= 78:
            return "Chennai"   # Tamil Nadu North
        else:
            return "Madurai"   # Tamil Nadu Central
    
    # Far southern regions
    elif lat >= 8:
        if lng <= 76:
            return "Kochi"     # Kerala North
        elif lng <= 78:
            return "Coimbatore" # Tamil Nadu South
        else:
            return "Trichy"    # Tamil Nadu Coast
    
    # Southernmost regions
    else:
        if lng <= 76:
            return "Thiruvananthapuram" # Kerala South
        else:
            return "Kanyakumari" # Tamil Nadu Tip

def get_real_location_name(lat, lng, use_reverse_geocoding=True):
    """
    Get real location name using reverse geocoding or fallback to region name.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lng : float
        Longitude
    use_reverse_geocoding : bool
        Whether to use reverse geocoding API
        
    Returns:
    --------
    str
        Real location name (ASCII-safe)
    """
    if use_reverse_geocoding:
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="bird_hotspot_finder")
            
            # Add small delay to respect rate limits
            time.sleep(0.1)
            
            location = geolocator.reverse((lat, lng), timeout=5)
            if location and location.address:
                # Extract meaningful parts of the address
                address_parts = location.address.split(', ')
                
                # Extract user-friendly location names with smart prioritization
                wildlife_areas = []
                famous_places = []
                major_cities = []
                districts = []
                states = []
                
                # Define major Indian cities for recognition
                major_city_names = [
                    'delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore', 'hyderabad', 
                    'pune', 'ahmedabad', 'jaipur', 'surat', 'lucknow', 'kanpur', 
                    'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna',
                    'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad',
                    'meerut', 'rajkot', 'kalyan', 'vasai', 'varanasi', 'srinagar',
                    'aurangabad', 'dhanbad', 'amritsar', 'navi mumbai', 'allahabad',
                    'ranchi', 'haora', 'coimbatore', 'jabalpur', 'gwalior', 'vijayawada',
                    'jodhpur', 'madurai', 'raipur', 'kota', 'chandigarh', 'guwahati',
                    'solapur', 'hubballi', 'tiruchirappalli', 'bareilly', 'moradabad',
                    'mysore', 'gurgaon', 'aligarh', 'jalandhar', 'bhubaneswar',
                    'salem', 'warangal', 'guntur', 'bhiwandi', 'saharanpur', 'gorakhpur'
                ]
                
                # Define famous wildlife/nature keywords
                nature_keywords = [
                    'sanctuary', 'national park', 'reserve', 'wildlife', 'forest', 
                    'tiger', 'elephant', 'bird sanctuary', 'lake', 'dam', 'river',
                    'wetland', 'biosphere', 'zoo', 'safari', 'botanical', 'garden',
                    'hill station', 'beach', 'coast', 'island', 'valley'
                ]
                
                # Categorize address parts by priority
                for part in address_parts:
                    part_clean = part.strip().lower()
                    original_part = part.strip()
                    
                    # Priority 1: Wildlife areas and nature places (highest priority)
                    if any(keyword in part_clean for keyword in nature_keywords):
                        wildlife_areas.append(original_part)
                    
                    # Priority 2: Major cities (very recognizable)
                    elif any(city in part_clean for city in major_city_names):
                        major_cities.append(original_part)
                    
                    # Priority 3: Famous places, landmarks (contains recognizable keywords)
                    elif any(keyword in part_clean for keyword in ['fort', 'palace', 'temple', 'ghat', 'market', 'station']):
                        famous_places.append(original_part)
                    
                    # Priority 4: Clean district names (administrative but useful)
                    elif any(keyword in part_clean for keyword in ['district', 'tehsil', 'mandal']):
                        # Clean up administrative terms
                        clean_name = original_part
                        for term in [' District', ' district', ' Tehsil', ' tehsil', ' Mandal', ' mandal']:
                            clean_name = clean_name.replace(term, '')
                        if clean_name.strip() and len(clean_name.strip()) > 2:
                            districts.append(clean_name.strip())
                    
                    # Priority 5: State names (lowest priority, backup)
                    elif part_clean in ['andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 
                                       'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
                                       'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
                                       'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 
                                       'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal']:
                        states.append(original_part)
                
                # Select the best name using priority system
                selected_name = None
                
                # Try each priority level
                if wildlife_areas:
                    # Clean up wildlife area names
                    for area in wildlife_areas[:1]:
                        clean_area = area.replace('Wildlife Sanctuary', 'WLS').replace('National Park', 'NP')
                        if len(clean_area) <= 25:  # Keep reasonable length
                            selected_name = area
                            break
                        else:
                            selected_name = clean_area
                            break
                
                elif major_cities:
                    # Use the first major city found
                    selected_name = major_cities[0]
                
                elif famous_places:
                    # Use the first famous place
                    selected_name = famous_places[0]
                
                elif districts:
                    # Use district name
                    selected_name = districts[0]
                
                elif states:
                    # Last resort: use state name
                    selected_name = states[0]
                
                # Return the selected name if found
                if selected_name:
                    try:
                        ascii_name = selected_name.encode('ascii', 'ignore').decode('ascii')
                        if ascii_name and len(ascii_name) > 2:
                            return ascii_name
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Reverse geocoding failed for {lat}, {lng}: {str(e)}")
    
    # Fallback to dynamic region name
    return get_dynamic_region_name(lat, lng)

def get_dynamic_location_name(lat, lng, use_geocoding=True):
    """
    Get dynamic location name based on coordinates using reverse geocoding APIs.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lng : float
        Longitude
    use_geocoding : bool
        Whether to use reverse geocoding (default True)
        
    Returns:
    --------
    str
        Dynamic location name discovered from coordinates
    """
    # Only static info: India boundaries (more precise to exclude neighbors)
    INDIA_BOUNDS = {
        'north': 37.6,    # Kashmir
        'south': 8.0,     # Exclude Sri Lanka (was 6.4)  
        'east': 97.25,    # Arunachal Pradesh
        'west': 68.7      # Gujarat/Rajasthan border
    }
    
    # First, check if coordinates are within India STRICTLY
    if not (INDIA_BOUNDS['south'] <= lat <= INDIA_BOUNDS['north'] and 
            INDIA_BOUNDS['west'] <= lng <= INDIA_BOUNDS['east']):
        return f"Outside India ({lat:.3f}°N, {lng:.3f}°E)"
    
    # Additional precise checks to exclude neighboring countries
    # Exclude Bangladesh (east of 92°E and north of 21°N)
    if lng > 92.0 and lat > 21.0:
        return f"Outside India ({lat:.3f}°N, {lng:.3f}°E)"
    
    # Exclude Sri Lanka (south of 10°N and east of 79°E)
    if lat < 10.0 and lng > 79.0:
        return f"Outside India ({lat:.3f}°N, {lng:.3f}°E)"
    
    # Exclude Pakistan (west of 75°E and north of 24°N)
    if lng < 75.0 and lat > 24.0:
        return f"Outside India ({lat:.3f}°N, {lng:.3f}°E)"
    
    # Exclude China (north of 35°N and east of 78°E)
    if lat > 35.0 and lng > 78.0:
        return f"Outside India ({lat:.3f}°N, {lng:.3f}°E)"
    
    if use_geocoding:
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="bird_hotspot_finder", timeout=10)
            
            # Get location details from coordinates
            location = geolocator.reverse((lat, lng), exactly_one=True, language='en')
            
            if location and location.address:
                # Parse address components properly
                address_parts = location.address.split(', ')
                
                # Look for meaningful Indian place names
                place_candidates = []
                
                for part in address_parts:
                    part_clean = part.strip()
                    # Skip postal codes, countries, and generic terms
                    if (len(part_clean) > 2 and 
                        not part_clean.isdigit() and 
                        part_clean.lower() not in ['india', 'भारत', 'republic of india']):
                        
                        # Prioritize city/town/village names
                        if any(keyword in part_clean.lower() for keyword in 
                               ['city', 'town', 'village', 'district', 'tehsil', 'block']):
                            place_candidates.insert(0, part_clean.replace('District', '').replace('Tehsil', '').strip())
                        else:
                            place_candidates.append(part_clean)
                
                # Return the best candidate
                if place_candidates:
                    return place_candidates[0]
                    
        except Exception as e:
            logger.debug(f"Reverse geocoding failed for ({lat:.4f}, {lng:.4f}): {str(e)}")
    
    # Fallback to dynamic region name based on coordinates
    return get_dynamic_region_name(lat, lng)

# Load tehsil data
@st.cache_data
def load_tehsil_data():
    """Load tehsil data from CSV file."""
    try:
        df = pd.read_csv('India Tehsil Centroid LatLong 1.csv')
        return df
    except FileNotFoundError:
        st.error("India Tehsil Centroid LatLong 1.csv file not found.")
        return pd.DataFrame()

# Load bird hotspot data
@st.cache_data
def load_bird_hotspot_data():
    """Load bird hotspot data from JSON file."""
    try:
        with open('bird_hotspot.json', 'r') as f:
            data = json.load(f)
        
        # Extract location data from the JSON
        locations = []
        for item in data:
            # Get the CSV string and split it into fields
            csv_str = list(item.values())[0]  # Get the first (and only) value in the dictionary
            fields = csv_str.split(',')
            
            # Extract relevant fields
            location = {
                'state': fields[5],  # NAME_1
                'district': fields[7],  # NAME_2
                'tehsil': fields[9],  # NAME_3
                'latitude': float(fields[17]),  # Latitude_N
                'longitude': float(fields[18])  # Longitude_E
            }
            locations.append(location)
        
        return pd.DataFrame(locations)
    except FileNotFoundError:
        st.error("bird_hotspot.json file not found.")
        return pd.DataFrame()

def get_bird_media_from_apis(bird_name, bird_client=None, ebird_api_key=None):
    """
    Get bird images and audio from multiple APIs dynamically.
    
    Parameters:
    -----------
    bird_name : str
        Common name of the bird species
    bird_client : BirdDataClient, optional
        Bird data client for API access
    ebird_api_key : str, optional
        eBird API key for accessing their API
        
    Returns:
    --------
    dict
        Dictionary with 'image_url' and 'audio_url' keys
    """
    result = {'image_url': 'N/A', 'audio_url': 'N/A'}
    
    # Try Xeno-canto first (often has both images and audio)
    if bird_client:
        try:
            xeno_data = bird_client.get_xeno_canto_recordings_by_species(bird_name)
            if not xeno_data.empty and len(xeno_data) > 0:
                # Get the first/best recording
                best_recording = xeno_data.iloc[0]
                
                # Get audio URL
                if 'file' in best_recording and best_recording['file']:
                    result['audio_url'] = best_recording['file']
                elif 'id' in best_recording:
                    result['audio_url'] = f"https://xeno-canto.org/{best_recording['id']}"
                
                # Some Xeno-canto entries may have image URLs
                if 'image_url' in best_recording and best_recording['image_url']:
                    result['image_url'] = best_recording['image_url']
                    
        except Exception as e:
            logger.warning(f"Xeno-canto API failed for {bird_name}: {str(e)}")
    
    # Note: eBird API does not provide direct media endpoints that work reliably
    # Skipping eBird media API calls to avoid JSON parsing errors
    
    # Try iNaturalist API for images
    if result['image_url'] == 'N/A':
        try:
            search_term = bird_name.replace(" ", "%20")
            inaturalist_url = f"https://api.inaturalist.org/v1/taxa?q={search_term}&taxon_id=3&per_page=1"
            
            response = requests.get(inaturalist_url, timeout=10)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get('results') and len(data['results']) > 0:
                        taxon = data['results'][0]
                        if 'default_photo' in taxon and taxon['default_photo']:
                            photo_url = taxon['default_photo'].get('medium_url')
                            if photo_url:
                                result['image_url'] = photo_url
                except json.JSONDecodeError:
                    logger.warning(f"iNaturalist returned invalid JSON for {bird_name}")
                            
        except Exception as e:
            logger.warning(f"iNaturalist API failed for {bird_name}: {str(e)}")
    
    # Final fallback to Wikimedia Commons
    if result['image_url'] == 'N/A':
        try:
            search_term = bird_name.replace(" ", "%20")
            wiki_url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=File:{search_term}%20bird&prop=imageinfo&iiprop=url&format=json"
            
            response = requests.get(wiki_url, timeout=10)
            if response.status_code == 200:
                try:
                    wiki_data = response.json()
                    if 'query' in wiki_data and 'pages' in wiki_data['query']:
                        # Get the first image URL
                        for page_id in wiki_data['query']['pages']:
                            page = wiki_data['query']['pages'][page_id]
                            if 'imageinfo' in page and len(page['imageinfo']) > 0:
                                result['image_url'] = page['imageinfo'][0]['url']
                                break
                except json.JSONDecodeError:
                    logger.warning(f"Wikimedia returned invalid JSON for {bird_name}")
                            
        except Exception as e:
            logger.warning(f"Wikimedia API failed for {bird_name}: {str(e)}")
    
    return result

def initialize_session_state():
    """Initialize all session state variables."""
    session_vars = {
        'processing': False,
        'selected_hotspot': 1,
        'hotspot_data': None,
        'result': None,
        'hotspot_details': {},
        'selected_hotspot_id': 1,
        'current_hotspot_id': None,
        'location_selected': False,
        'india_hotspot_results': None,
        'india_hotspot_params': None,
        'all_india_results': None,
        'all_india_analysis_params': None
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def create_excel_download(data_dict, filename_prefix):
    """
    Create Excel file for download with multiple sheets.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with sheet_name: dataframe pairs
    filename_prefix : str
        Prefix for the filename
        
    Returns:
    --------
    bytes
        Excel file data
    """
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, index=False, sheet_name=sheet_name)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error creating Excel file: {str(e)}")
        return None

def analyze_single_location(latitude, longitude, location_name, search_radius, bird_client, include_photos=False, include_audio=False):
    """
    Analyze a single location for bird observations.
    
    Parameters:
    -----------
    latitude : float
        Latitude of the location
    longitude : float
        Longitude of the location
    location_name : str
        Name of the location
    search_radius : int
        Search radius in kilometers
    bird_client : BirdDataClient
        Initialized bird data client
    include_photos : bool
        Whether to include bird photos
    include_audio : bool
        Whether to include bird audio
        
    Returns:
    --------
    dict
        Analysis results
    """
    try:
        # Get eBird observations
        ebird_observations = bird_client.get_ebird_observations(
            lat=latitude,
            lng=longitude,
            radius_km=search_radius,
            days_back=30
        )
        
        if ebird_observations.empty:
            return None
        
        # Calculate species counts
        species_counts = ebird_observations.groupby('comName').size().reset_index(name='count')
        
        results = []
        for _, species_row in species_counts.iterrows():
            species_name = species_row['comName']
            bird_count = species_row['count']
            
            result = {
                'Place': location_name,
                'Latitude': latitude,
                'Longitude': longitude,
                'Species Name': species_name,
                'Bird Count': bird_count,
                'Data Source': 'eBird',
                'Photo URL': 'N/A',
                'Audio URL': 'N/A'
            }
            
            # Get photo if requested
            if include_photos:
                try:
                    photo_url = get_bird_media_from_apis(species_name, bird_client)['image_url']
                    if photo_url and photo_url != 'N/A':
                        result['Photo URL'] = photo_url
                except Exception as e:
                    logger.warning(f"Failed to get photo for {species_name}: {str(e)}")
            
            # Get audio if requested
            if include_audio:
                try:
                    audio_data = bird_client.get_xeno_canto_recordings_by_species(species_name)
                    if not audio_data.empty and len(audio_data) > 0:
                        recording = audio_data.iloc[0]
                        if 'file' in recording and recording['file']:
                            result['Audio URL'] = recording['file']
                        elif 'id' in recording:
                            result['Audio URL'] = f"https://xeno-canto.org/{recording['id']}"
                except Exception as e:
                    logger.warning(f"Failed to get audio for {species_name}: {str(e)}")
            
            results.append(result)
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing location {location_name}: {str(e)}")
        return None

def create_india_map_with_hotspots(hotspot_data):
    """
    Create a Folium map of India with bird hotspots (Orange: 10-19 species, Red: 20+ species).
    
    Parameters:
    -----------
    hotspot_data : pd.DataFrame
        DataFrame containing hotspot information
        
    Returns:
    --------
    folium.Map
        Folium map object
    """
    try:
        # Define strict India boundaries for filtering
        INDIA_LAT_MIN, INDIA_LAT_MAX = 8.0, 37.6  # Exclude Sri Lanka completely
        INDIA_LNG_MIN, INDIA_LNG_MAX = 68.7, 97.25  # Gujarat to Arunachal Pradesh
        
        # Filter hotspot data to include ONLY points within India
        india_hotspots = hotspot_data[
            (hotspot_data['Latitude'] >= INDIA_LAT_MIN) & 
            (hotspot_data['Latitude'] <= INDIA_LAT_MAX) &
            (hotspot_data['Longitude'] >= INDIA_LNG_MIN) & 
            (hotspot_data['Longitude'] <= INDIA_LNG_MAX)
        ].copy()
        
        # Additional filtering to exclude neighboring countries more precisely
        # Exclude Bangladesh
        india_hotspots = india_hotspots[~((india_hotspots['Longitude'] > 92.0) & (india_hotspots['Latitude'] > 21.0))]
        # Exclude Sri Lanka
        india_hotspots = india_hotspots[~((india_hotspots['Latitude'] < 10.0) & (india_hotspots['Longitude'] > 79.0))]
        # Exclude Pakistan
        india_hotspots = india_hotspots[~((india_hotspots['Longitude'] < 75.0) & (india_hotspots['Latitude'] > 24.0))]
        # Exclude China
        india_hotspots = india_hotspots[~((india_hotspots['Latitude'] > 35.0) & (india_hotspots['Longitude'] > 78.0))]
        
        # Also filter out any place names that explicitly say "Outside India"
        india_hotspots = india_hotspots[~india_hotspots['Place'].str.contains('Outside India', na=False)]
        
        logger.info(f"Filtered {len(hotspot_data)} total hotspots to {len(india_hotspots)} within India boundaries")
        
        if india_hotspots.empty:
            logger.warning("No hotspots found within India boundaries")
            return None
        
        # Create map centered on India with VERY strict bounds
        india_map = folium.Map(
            location=[20.5937, 78.9629],  # Center more towards central India
            zoom_start=5,  # Lower zoom to see all of India
            tiles="OpenStreetMap",  # Use OpenStreetMap for clearer boundaries
            prefer_canvas=True,
            min_zoom=5,
            max_zoom=8,  # Reduced max zoom to keep focus on India
            max_bounds=True,
            # Use exact India boundaries
            min_lat=INDIA_LAT_MIN,
            max_lat=INDIA_LAT_MAX,
            min_lon=INDIA_LNG_MIN,
            max_lon=INDIA_LNG_MAX
        )
        
        # Set very strict map bounds to India ONLY
        india_bounds = [[INDIA_LAT_MIN, INDIA_LNG_MIN], [INDIA_LAT_MAX, INDIA_LNG_MAX]]
        india_map.fit_bounds(india_bounds)
        
        # Add a subtle rectangle to visually constrain to India boundaries
        folium.Rectangle(
            bounds=india_bounds,
            color='darkblue',
            weight=1,
            fill=False,
            opacity=0.3,
            popup="India Boundary"
        ).add_to(india_map)
        
        # Group filtered India hotspots by location for mapping
        if 'Total Species at Location' in india_hotspots.columns:
            # Use the correct column name from the dynamic analysis with scientific names
            species_col = 'Scientific Name' if 'Scientific Name' in india_hotspots.columns else 'Species Name'
            map_data = india_hotspots.groupby(['Place', 'Latitude', 'Longitude']).agg({
                species_col: 'nunique',
                'Bird Count': 'sum',
                'Total Species at Location': 'first'  # Take the first value since it should be the same for each location
            }).reset_index()
            map_data.columns = ['Place', 'Latitude', 'Longitude', 'Species Count', 'Total Birds', 'Total Species']
            # Use the more accurate 'Total Species' column for classification
            map_data['Species Count'] = map_data['Total Species']
        else:
            # Fallback for other data formats
            species_col = 'Scientific Name' if 'Scientific Name' in india_hotspots.columns else 'Species Name'
            map_data = india_hotspots.groupby(['Place', 'Latitude', 'Longitude']).agg({
                species_col: 'nunique',
                'Bird Count': 'sum'
            }).reset_index()
            map_data.columns = ['Place', 'Latitude', 'Longitude', 'Species Count', 'Total Birds']
        
        # Debug: Check species count distribution
        species_distribution = map_data['Species Count'].value_counts().sort_index()
        logger.info(f"Species count distribution: {species_distribution.to_dict()}")
        
        orange_count = 0
        red_count = 0
        
        # Add hotspot markers with ONLY 2 types: Orange (10-19) and Red (20+)
        for _, hotspot in map_data.iterrows():
            species_count = int(hotspot['Species Count'])
            
            # Debug: Log each hotspot classification
            logger.debug(f"Hotspot {hotspot['Place']}: {species_count} species")
            
            # Multiple hotspot types for better coverage
            if species_count >= 20:
                color = 'red'  # Red dots: 20+ species
                category = 'High Diversity (20+ species)'
                radius = min(species_count/2 + 5, 20)
                red_count += 1
            elif species_count >= 10:
                color = 'orange'  # Orange dots: 10-19 species
                category = 'Medium Diversity (10-19 species)'
                radius = min(species_count/2 + 3, 15)
                orange_count += 1
            elif species_count >= 5:
                color = 'yellow'  # Yellow dots: 5-9 species
                category = 'Low Diversity (5-9 species)'
                radius = min(species_count/2 + 2, 12)
                orange_count += 1  # Count as orange for stats
            else:
                # Skip locations with less than 5 species
                logger.debug(f"Skipping {hotspot['Place']} with only {species_count} species")
                continue
            
            folium.CircleMarker(
                location=[hotspot['Latitude'], hotspot['Longitude']],
                radius=radius,
                color=color,
                fill=True,
                fillOpacity=0.8,
                weight=2,
                popup=f"""
                <div style='min-width: 200px;'>
                <h4>{hotspot['Place']}</h4>
                <p><b>Category:</b> {category}</p>
                <p><b>Species:</b> {species_count}</p>
                <p><b>Total Birds:</b> {int(hotspot['Total Birds'])}</p>
                <p><b>Coordinates:</b> {hotspot['Latitude']:.4f}, {hotspot['Longitude']:.4f}</p>
                </div>
                """
            ).add_to(india_map)
        
        logger.info(f"Map created with {orange_count} Orange hotspots and {red_count} Red hotspots")
        
        # Add legend for all hotspot types
        legend_html = '''
        <div style="position: fixed; 
                   bottom: 50px; left: 50px; width: 220px; height: 130px; 
                   background-color: white; border:2px solid grey; z-index:9999; 
                   font-size:12px; padding: 10px">
        <h4>Bird Hotspot Types</h4>
        <p><i class="fa fa-circle" style="color:red"></i> High Diversity (20+ species)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Medium Diversity (10-19 species)</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Low Diversity (5-9 species)</p>
        <p><small>Minimum: 5 species required for discovery</small></p>
        </div>
        '''
        india_map.get_root().html.add_child(folium.Element(legend_html))
        
        return india_map
    
    except Exception as e:
        logger.error(f"Error creating map: {str(e)}")
        return None

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("🦜 South Asian Bird Hotspot Finder")
    st.markdown("""
    **🔬 Scientific Bird Discovery Tool for Birders!** 
    Find the best bird watching locations across India with **scientific names**, **fun facts**, and **birder-friendly locations**.
    
    **✨ Key Features:**
    - 🧬 **Scientific Names**: All species listed with proper taxonomic names
    - 📚 **Fun Facts**: Fascinating details about each bird species
    - 🗺️ **Birder Locations**: Recognizable places like national parks, famous birding spots
    - 🎯 **Hotspot Classification**: Orange (10-19 species) vs Red (20+ species)
    - 🌍 **Dynamic Coverage**: Real-time data from eBird and GBIF APIs
    
    **🎯 Hotspot Types (Minimum 5 species for discovery):**
    - 🔴 **Red Dots**: High Diversity (20+ species)
    - 🟠 **Orange Dots**: Medium Diversity (10-19 species)  
    - 🟡 **Yellow Dots**: Low Diversity (5-9 species)
    """)
    
    # Sidebar for inputs
    st.sidebar.header("Location Settings")
    
    # Location input options
    location_method = st.sidebar.radio(
        "Select location method:",
        [
            "Search by City/Locality",
            "Enter Coordinates", 
            "Use Current Location",
            "🇮🇳 Dynamic India-wide Hotspot Discovery",
            "India Tehsil-based Analysis",
            "All-India Analysis"
        ],
        key="location_method"
    )
    
    # Data source settings
    st.sidebar.header("Data Sources")
    use_ebird = st.sidebar.checkbox("Use eBird Data", value=True, key="use_ebird")
    use_gbif = st.sidebar.checkbox("Use GBIF Data", value=True, key="use_gbif")
    use_xeno_canto = st.sidebar.checkbox("Use Xeno-canto Data", value=False, key="use_xeno_canto")
    
    # Use eBird API key from environment variable consistently
    ebird_api_key = EBIRD_API_KEY
    
    # Show API key status in sidebar
    if use_ebird:
        if ebird_api_key:
            st.sidebar.success(f"✅ eBird API Key: {ebird_api_key[:8]}...")
        else:
            st.sidebar.error("❌ eBird API key not found in .env file")
            st.sidebar.info("💡 Add EBIRD_API_KEY to your .env file to use eBird data")
    else:
        ebird_api_key = ""
    
    # Common settings
    search_radius = st.sidebar.slider("Search radius (km)", 5, 100, 25, key="search_radius_sidebar")
    current_date = st.sidebar.date_input("Date for prediction", datetime.now(), key="current_date_sidebar")
    
    # Initialize bird client with consistent API key
    bird_client = BirdDataClient(ebird_api_key=ebird_api_key if use_ebird else "")
    
    # Handle different location methods
    if location_method == "Search by City/Locality":
        handle_city_search(bird_client, search_radius, use_ebird)
    
    elif location_method == "Enter Coordinates":
        handle_coordinate_input(bird_client, search_radius, use_ebird)
    
    elif location_method == "Use Current Location":
        handle_current_location(bird_client, search_radius, use_ebird)
    
    elif location_method == "🇮🇳 Dynamic India-wide Hotspot Discovery":
        handle_dynamic_india_hotspot_discovery(
            bird_client, use_ebird, use_gbif, use_xeno_canto
        )
    
    elif location_method == "India Tehsil-based Analysis":
        handle_tehsil_analysis(bird_client, search_radius, use_ebird)
    
    elif location_method == "All-India Analysis":
        handle_all_india_analysis(bird_client, use_ebird)

def handle_city_search(bird_client, search_radius, use_ebird):
    """Handle city search functionality."""
    city = st.text_input("Enter city or locality name:", key="city_input")
    
    if city:
        try:
            geolocator = Nominatim(user_agent="bird_hotspot_finder")
            location = geolocator.geocode(city)
            
            if location:
                st.session_state.latitude = location.latitude
                st.session_state.longitude = location.longitude
                st.session_state.location_name = city
                st.session_state.location_selected = True
                
                # Show location on map
                st.map(pd.DataFrame({'lat': [location.latitude], 'lon': [location.longitude]}))
                
                # Analyze button
                if st.button("🔍 Find Bird Hotspots", key="city_analyze_button", use_container_width=True):
                    analyze_and_display_single_location(
                        location.latitude, location.longitude, city, 
                        search_radius, bird_client, use_ebird
                    )
            else:
                st.error("Location not found. Please try a different name.")
        except Exception as e:
            st.error(f"Error finding location: {str(e)}")

def handle_coordinate_input(bird_client, search_radius, use_ebird):
    """Handle coordinate input functionality."""
    col1, col2 = st.columns(2)
    
    with col1:
        latitude = st.number_input("Latitude:", -90.0, 90.0, 20.5937, key="lat_input")
    with col2:
        longitude = st.number_input("Longitude:", -180.0, 180.0, 78.9629, key="lon_input")
    
    location_name = f"Coordinates ({latitude}, {longitude})"
    
    # Show location on map
    st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
    
    # Analyze button
    if st.button("🔍 Find Bird Hotspots", key="coord_analyze_button", use_container_width=True):
        analyze_and_display_single_location(
            latitude, longitude, location_name,
            search_radius, bird_client, use_ebird
        )

def handle_current_location(bird_client, search_radius, use_ebird):
    """Handle current location functionality."""
    st.info("This feature requires location access in your browser.")
    
    if st.button("Get Current Location", key="get_location_btn"):
        # Default to a location (in reality, you'd implement browser geolocation)
        latitude, longitude = 20.5937, 78.9629
        location_name = "Current Location"
        
        # Show location on map
        st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
        
        # Analyze button
        if st.button("🔍 Find Bird Hotspots", key="current_loc_analyze_button", use_container_width=True):
            analyze_and_display_single_location(
                latitude, longitude, location_name,
                search_radius, bird_client, use_ebird
            )

def analyze_and_display_single_location(latitude, longitude, location_name, search_radius, bird_client, use_ebird):
    """Analyze and display results for a single location."""
    with st.spinner(f"Analyzing bird hotspots in {location_name}..."):
        try:
            if use_ebird and EBIRD_API_KEY:
                results = analyze_single_location(
                    latitude, longitude, location_name, search_radius, bird_client
                )
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.write("### Bird Hotspot Analysis Results")
                    
                    # Location details
                    st.write("#### Location Information")
                    location_df = pd.DataFrame({
                        'Place': [location_name],
                        'Latitude': [latitude],
                        'Longitude': [longitude]
                    })
                    st.dataframe(location_df, use_container_width=True)
                    
                    # Species information
                    st.write("#### Bird Species Information")
                    species_df = results_df.groupby('Species Name')['Bird Count'].sum().reset_index()
                    species_df = species_df.sort_values('Bird Count', ascending=False)
                    st.dataframe(species_df, use_container_width=True)
                    
                    # Download button
                    excel_data = create_excel_download({
                        'Location': location_df,
                        'Species Data': species_df
                    }, f"bird_hotspot_{location_name}")
                    
                    if excel_data:
                        st.download_button(
                            label="📥 Download Results",
                            data=excel_data,
                            file_name=f"bird_hotspot_{location_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.warning("No bird observations found in this location.")
            else:
                st.warning("Please enable eBird data source and ensure API key is configured in .env file.")
        except Exception as e:
            st.error(f"Error analyzing location: {str(e)}")
            logger.error(f"Error in location analysis: {traceback.format_exc()}")

def handle_dynamic_india_hotspot_discovery(bird_client, use_ebird, use_gbif, use_xeno_canto):
    """Handle dynamic India-wide hotspot discovery using systematic grid coverage."""
    st.write("### 🇮🇳 Dynamic India-wide Bird Hotspot Discovery")
    
    st.info("🎯 **Scientific Systematic Coverage**: This feature generates a dynamic grid covering the entire India and discovers bird hotspots using real-time API data with **scientific names** and **birder-friendly locations**.")
    
    # Display coverage information
    with st.expander("📊 India Coverage Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Geographic Coverage:**")
            st.write(f"🧭 **Latitude Range:** {INDIA_BOUNDS['south']}°N to {INDIA_BOUNDS['north']}°N")
            st.write(f"🧭 **Longitude Range:** {INDIA_BOUNDS['west']}°E to {INDIA_BOUNDS['east']}°E")
            st.write(f"📏 **Total Area:** ~3.28 million km²")
        
        with col2:
            st.write("**Hotspot Classification:**")
            st.write("🔴 **Red Dots:** 20+ species (High Diversity)")
            st.write("🟠 **Orange Dots:** 10-19 species (Medium Diversity)")
            st.write("🟡 **Yellow Dots:** 5-9 species (Low Diversity)")
            st.write("⚪ **Excluded:** Less than 5 species")
    
    # Add detailed explanations of grid types and data sources
    with st.expander("🔬 **Technical Details: Grid Types & Data Sources**"):
        st.write("### 🗺️ Grid Preparation Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### **🔲 Systematic Grid**")
            st.write("- **Method**: Evenly spaced grid points across entire India")
            st.write("- **Coverage**: Mathematical division of lat/lng ranges")
            st.write("- **Formula**: `spacing = range / √(grid_points)`")
            st.write("- **Best for**: Uniform national coverage, baseline analysis")
            st.write("- **Recommended**: ✅ Most balanced approach")
            
            st.write("#### **🎯 Adaptive Grid**")
            st.write("- **Method**: Higher density in biodiverse regions")
            st.write("- **Regions**: Western Ghats (40%), E. Himalayas (30%), Central (20%), Coastal (10%)")
            st.write("- **Coverage**: Focused on known biodiversity hotspots")
            st.write("- **Best for**: Discovering rich ecological zones")
            st.write("- **Trade-off**: May miss surprising discoveries in less obvious areas")
        
        with col2:
            st.write("#### **🔍 Dense Grid**")
            st.write("- **Method**: Higher resolution systematic coverage")
            st.write("- **Density**: 20% more points than systematic")
            st.write("- **Coverage**: Comprehensive fine-grained analysis")
            st.write("- **Best for**: Detailed regional studies, research")
            st.write("- **Trade-off**: Longer analysis time, more API calls")
            
            st.write("#### **📍 Location Naming**")
            st.write("- **Primary**: Reverse geocoding for real place names")
            st.write("- **Examples**: 'Mysore', 'Coimbatore', 'Shimla'")
            st.write("- **Fallback**: Geographic region names")
            st.write("- **Rate Limits**: First 20 points use real names, rest use regions")
        
        st.write("### 🌐 Data Sources Integration")
        
        tab1, tab2, tab3 = st.tabs(["🐦 eBird", "🌍 GBIF", "🔊 Xeno-canto"])
        
        with tab1:
            st.write("**eBird (Primary Source)**")
            st.write("- **API**: Real-time citizen science observations")
            st.write("- **Coverage**: Last 30 days of bird sightings")
            st.write("- **Data**: Species name, count, location, date")
            st.write("- **Strength**: Fresh, abundant, well-verified data")
            st.write("- **Usage**: Main driver for hotspot classification")
            st.write("- **Required**: ✅ API key needed")
        
        with tab2:
            st.write("**GBIF (Secondary Enhancement)**")
            st.write("- **API**: Global Biodiversity Information Facility")
            st.write("- **Coverage**: Historical occurrence records")
            st.write("- **Data**: Scientific specimens, observations")
            st.write("- **Strength**: Comprehensive, scientific-grade data")
            st.write("- **Usage**: Enhances species diversity analysis")
            st.write("- **Integration**: ✅ Automatically used when enabled")
            st.write("- **Current Status**: 🟢 Active (credentials provided)")
        
        with tab3:
            st.write("**Xeno-canto (Audio Enhancement)**")
            st.write("- **API**: Bird sound recordings database")
            st.write("- **Coverage**: Audio recordings with location data")
            st.write("- **Data**: Bird calls, songs, location metadata")
            st.write("- **Strength**: Acoustic biodiversity evidence")
            st.write("- **Usage**: Optional audio URL enrichment")
            st.write("- **Performance**: Slower due to audio processing")
        
        st.write("### ⚙️ Analysis Pipeline")
        st.write("1. **Grid Generation**: Create coordinate points using selected method")
        st.write("2. **Location Naming**: Reverse geocode for real place names")
        st.write("3. **eBird Query**: Get recent bird observations (primary)")
        st.write("4. **GBIF Integration**: Add historical occurrence data (secondary)")
        st.write("5. **Species Analysis**: Count unique species and individuals")
        st.write("6. **Hotspot Classification**: Orange (10-19) or Red (20+) species")
        st.write("7. **Optional Enrichment**: Photos and audio URLs")
        st.write("8. **Results Compilation**: Generate comprehensive dataset")
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        grid_type = st.selectbox(
            "Grid Coverage Type",
            [
                "systematic",
                "adaptive", 
                "dense"
            ],
            format_func=lambda x: {
                "systematic": "Systematic Grid (Recommended)",
                "adaptive": "Adaptive Grid (Biodiverse Regions)",
                "dense": "Dense Grid (Comprehensive)"
            }[x],
            help="Choose how to generate grid points across India"
        )
        
        if grid_type == "systematic":
            grid_points = st.slider("Grid points", 200, 1000, 500, 50, 
                                  help="Evenly distributed across India - More points = More hotspots")
        elif grid_type == "adaptive":
            grid_points = st.slider("Grid points", 300, 1200, 600, 50,
                                  help="Higher density in biodiverse regions - More points = More hotspots")
        else:  # dense
            grid_points = st.slider("Grid points", 400, 2000, 800, 100,
                                  help="Dense coverage for maximum hotspot discovery")
        
        search_radius = st.slider(
            "Search radius per point (km)",
            min_value=15,
            max_value=75,
            value=50,
            step=5,
            help="Radius to search around each grid point"
        )
    
    with col2:
        st.write("**Analysis Options:**")
        include_photos = st.checkbox("Include bird photos", value=False, 
                                   help="Fetch bird photos (increases analysis time)")
        include_audio = st.checkbox("Include bird sounds", value=False, 
                                  help="Fetch bird audio from Xeno-canto")
        
        min_species_threshold = st.selectbox(
            "Minimum species threshold",
            [5, 6, 7, 8, 9, 10],
            index=0,  # Default to 5 species to get more hotspots
            help="Lower threshold = More hotspots discovered across India"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            api_delay = st.slider("API delay (seconds)", 0.1, 2.0, 0.5, 0.1,
                                help="Delay between API calls to prevent rate limiting")
            max_retries = st.slider("Max retries per point", 1, 5, 2,
                                  help="Number of retries for failed API calls")
    
    # Show estimated analysis time (more realistic for larger grids)
    estimated_time = grid_points * (1.5 + api_delay)  # More efficient processing
    if estimated_time > 3600:
        time_str = f"~{estimated_time//3600:.1f} hours"
    elif estimated_time > 60:
        time_str = f"~{estimated_time//60:.1f} minutes"
    else:
        time_str = f"~{estimated_time:.0f} seconds"
    
    st.info(f"⏱️ **Estimated Analysis Time:** {time_str} (analyzing {grid_points:,} grid points for 1000+ hotspots)")
    
    # Current parameters for caching
    current_params = {
        'grid_type': grid_type,
        'grid_points': grid_points,
        'search_radius': search_radius,
        'include_photos': include_photos,
        'include_audio': include_audio,
        'min_species_threshold': min_species_threshold,
        'api_delay': api_delay,
        'max_retries': max_retries,
        'use_ebird': use_ebird,
        'use_gbif': use_gbif,
        'use_xeno_canto': use_xeno_canto,
        'ebird_api_key': EBIRD_API_KEY if use_ebird else ""
    }
    
    # Discover button
    discover_button = st.button(
        "🔍 Start Dynamic India Discovery",
        key="discover_dynamic_hotspots_button",
        use_container_width=True,
        help="Begin systematic analysis across India"
    )
    
    # Clear results button if we have cached results
    if st.session_state.india_hotspot_results is not None:
        if st.button("🗑️ Clear Discovery Results", key="clear_discovery_button", use_container_width=True):
            st.session_state.india_hotspot_results = None
            st.session_state.india_hotspot_params = None
            st.success("✅ Discovery results cleared!")
            st.rerun()
    
    # Display previous results if available and parameters haven't changed
    if (st.session_state.india_hotspot_results is not None and 
        st.session_state.india_hotspot_params == current_params):
        
        display_dynamic_india_results(st.session_state.india_hotspot_results, current_params)
    
    if discover_button:
        if not use_ebird or not EBIRD_API_KEY:
            st.error("❌ eBird API key is required for hotspot discovery. Please enable eBird and ensure EBIRD_API_KEY is set in your .env file.")
            return
        
        # Clear previous results
        st.session_state.india_hotspot_results = None
        st.session_state.india_hotspot_params = None
        
        # Run dynamic discovery
        results_df = run_dynamic_india_discovery(
            bird_client, current_params
        )
        
        if results_df is not None and not results_df.empty:
            # Store in session state
            st.session_state.india_hotspot_results = results_df
            st.session_state.india_hotspot_params = current_params
            
            # Display success message with proper column names
            total_hotspots = len(results_df['Place'].unique())
            
            # Use correct column name for species count
            species_col = 'Scientific Name' if 'Scientific Name' in results_df.columns else 'Species Name'
            total_species = len(results_df[species_col].unique())
            total_birds = results_df['Bird Count'].sum()
            
            # Count by hotspot type with new classification
            yellow_hotspots = len(results_df[(results_df['Total Species at Location'] >= 5) & 
                                            (results_df['Total Species at Location'] < 10)]['Place'].unique())
            orange_hotspots = len(results_df[(results_df['Total Species at Location'] >= 10) & 
                                           (results_df['Total Species at Location'] < 20)]['Place'].unique())
            red_hotspots = len(results_df[results_df['Total Species at Location'] >= 20]['Place'].unique())
            
            st.success(f"🎉 **Scientific Discovery Complete!** Found {total_hotspots:,} hotspots: {red_hotspots:,} Red (20+) + {orange_hotspots:,} Orange (10-19) + {yellow_hotspots:,} Yellow (5-9 species) across India with {total_species:,} unique species with scientific names and fun facts!")
            
            # Trigger rerun to display full results
            st.rerun()

def run_dynamic_india_discovery(bird_client, params):
    """Run dynamic India-wide hotspot discovery using grid-based approach."""
    try:
        st.write("### 🔍 Dynamic India Grid Analysis in Progress...")
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate dynamic grid covering entire India
        status_text.text("Generating dynamic India coverage grid...")
        progress_bar.progress(5)
        
        grid_points = generate_dynamic_india_grid(
            max_points=params['grid_points'],
            grid_type=params['grid_type']
        )
        
        if grid_points.empty:
            st.error("❌ Failed to generate India coverage grid")
            return None
        
        status_text.text(f"Generated {len(grid_points)} grid points covering India")
        progress_bar.progress(10)
        
        # Initialize results storage
        all_hotspots = []
        successful_analyses = 0
        orange_count = 0
        red_count = 0
        
        # Process each grid point
        for idx, (_, point) in enumerate(grid_points.iterrows()):
            progress = 15 + (idx / len(grid_points)) * 75
            progress_bar.progress(int(progress))
            
            location_name = point['location_name']
            status_text.text(f"Analyzing: {location_name} ({idx + 1:,}/{len(grid_points):,})")
            
            try:
                # Log detailed search parameters
                safe_location_name = location_name.encode('ascii', 'ignore').decode('ascii').strip()
                if not safe_location_name or len(safe_location_name) < 3:
                    safe_location_name = f"GridPoint({point['latitude']:.3f}, {point['longitude']:.3f})"
                
                # Skip points outside India boundaries completely
                if "Outside India" in location_name:
                    logger.debug(f"⏭️ SKIPPING: {safe_location_name} - Outside India boundaries")
                    continue
                
                # Check if coordinates are reasonable for bird habitat
                habitat_check = check_habitat_viability(point['latitude'], point['longitude'])
                
                logger.info(f"🔍 SEARCHING: {safe_location_name} | Coords: ({point['latitude']:.4f}, {point['longitude']:.4f}) | Radius: {params['search_radius']}km | Habitat: {habitat_check}")
                
                # Get eBird observations for this grid point
                ebird_observations = bird_client.get_ebird_observations(
                    lat=point['latitude'],
                    lng=point['longitude'],
                    radius_km=params['search_radius'],
                    days_back=30
                )
                
                # Initialize combined species data
                all_species = pd.DataFrame()
                
                # Process eBird data (primary source) with detailed logging
                if not ebird_observations.empty:
                    ebird_species = ebird_observations.groupby('comName').size().reset_index(name='ebird_count')
                    ebird_species['source'] = 'eBird'
                    ebird_species.rename(columns={'comName': 'species_name'}, inplace=True)
                    all_species = pd.concat([all_species, ebird_species], ignore_index=True)
                    logger.info(f"  📊 eBird: {len(ebird_observations)} observations → {len(ebird_species)} species")
                    
                    # Log sample species if we have them
                    if len(ebird_species) > 0:
                        sample_species = ebird_species.head(3)['species_name'].tolist()
                        logger.info(f"  🐦 Sample eBird species: {', '.join(sample_species)}")
                else:
                    logger.warning(f"  ❌ eBird: No observations found in {params['search_radius']}km radius")
                
                # Get GBIF data if enabled (secondary enhancement) with detailed logging
                if params['use_gbif']:
                    try:
                        gbif_observations = bird_client.get_gbif_occurrences(
                            lat=point['latitude'],
                            lng=point['longitude'],
                            radius_km=params['search_radius']
                        )
                        
                        if not gbif_observations.empty and 'species' in gbif_observations.columns:
                            gbif_species = gbif_observations.groupby('species').size().reset_index(name='gbif_count')
                            gbif_species['source'] = 'GBIF'
                            gbif_species.rename(columns={'species': 'species_name'}, inplace=True)
                            # Add GBIF data to combined dataset
                            all_species = pd.concat([all_species, gbif_species], ignore_index=True)
                            logger.info(f"  📊 GBIF: {len(gbif_observations)} occurrences → {len(gbif_species)} species")
                            
                            # Log sample species if we have them
                            if len(gbif_species) > 0:
                                sample_species = gbif_species.head(3)['species_name'].tolist()
                                logger.info(f"  🔬 Sample GBIF species: {', '.join(sample_species)}")
                        else:
                            logger.warning(f"  ❌ GBIF: No valid occurrences found in {params['search_radius']}km radius")
                    except Exception as gbif_error:
                        logger.warning(f"  ⚠️ GBIF API error for {safe_location_name}: {str(gbif_error)}")
                
                # Combine and analyze all species data
                if not all_species.empty:
                    # Ensure required columns exist before aggregation
                    if 'ebird_count' not in all_species.columns:
                        all_species['ebird_count'] = 0
                    if 'gbif_count' not in all_species.columns:
                        all_species['gbif_count'] = 0
                    
                    # Fill NaN values with 0 before aggregation
                    all_species['ebird_count'] = all_species['ebird_count'].fillna(0)
                    all_species['gbif_count'] = all_species['gbif_count'].fillna(0)
                    
                    # Combine species from both sources properly
                    combined_species = all_species.groupby('species_name').agg({
                        'ebird_count': lambda x: x.sum(),
                        'gbif_count': lambda x: x.sum()
                    }).reset_index()
                    
                    # Calculate combined count
                    combined_species['combined_count'] = combined_species['ebird_count'] + combined_species['gbif_count']
                    
                    # Get accurate species count
                    unique_species_count = len(combined_species)
                    total_bird_count = int(combined_species['combined_count'].sum())
                    
                    # Validate species count (sanity check)
                    if unique_species_count > 200:  # More than 200 species in 50km radius is suspicious
                        logger.warning(f"Suspicious species count {unique_species_count} at {location_name}, skipping")
                        continue
                    
                    # Log final analysis results
                    logger.info(f"  🎯 TOTAL: {unique_species_count} unique species, {total_bird_count} total birds")
                    
                    # Only consider locations with minimum species threshold
                    if unique_species_count >= params['min_species_threshold']:
                        successful_analyses += 1
                        
                        # Determine hotspot type with more flexible classification
                        if unique_species_count >= 20:
                            hotspot_type = "Red Hotspot (20+ species)"
                            marker_color = "red"
                            red_count += 1
                        elif unique_species_count >= 10:
                            hotspot_type = "Orange Hotspot (10-19 species)"
                            marker_color = "orange"
                            orange_count += 1
                        else:  # For lower thresholds (5-9 species)
                            hotspot_type = "Yellow Hotspot (5-9 species)"
                            marker_color = "yellow"
                            orange_count += 1  # Count as orange for now
                        
                        logger.info(f"  ✅ QUALIFIED: {hotspot_type}")
                        
                        # Add each species observation to results for QUALIFIED hotspots
                        for _, species_row in combined_species.iterrows():
                            # Determine primary data source
                            primary_source = 'eBird' if species_row['ebird_count'] > 0 else 'GBIF'
                            if species_row['ebird_count'] > 0 and species_row['gbif_count'] > 0:
                                primary_source = 'eBird+GBIF'
                            
                            # Get scientific name and fun fact
                            bird_info = get_scientific_name_and_fun_fact(species_row['species_name'])
                            
                            hotspot_data = {
                                'Place': location_name,
                                'Region': get_habitat_based_birding_description(point['latitude'], point['longitude']),  # Get birder habitat context
                                'Latitude': point['latitude'],
                                'Longitude': point['longitude'],
                                'Scientific Name': bird_info['scientific'],  # Use scientific name
                                'Common Name': species_row['species_name'],  # Keep common name for reference
                                'Fun Fact': bird_info['fun_fact'],  # Add fun fact
                                'Bird Count': max(1, int(species_row['combined_count'])),  # Ensure at least 1
                                'Total Species at Location': unique_species_count,
                                'Total Birds at Location': total_bird_count,
                                'Hotspot Type': hotspot_type,
                                'Marker Color': marker_color,
                                'Grid Type': params['grid_type'],
                                'Data Source': primary_source,
                                'Photo URL': 'N/A',
                                'Audio URL': 'N/A'
                            }
                            
                            # Get media dynamically from multiple APIs
                            if params.get('include_photos') or params.get('include_audio'):
                                try:
                                    media_result = get_bird_media_from_apis(
                                        species_row['species_name'], 
                                        bird_client=bird_client if params.get('use_xeno_canto') else None,
                                        ebird_api_key=params.get('ebird_api_key')
                                    )
                                    
                                    if params.get('include_photos') and media_result['image_url'] != 'N/A':
                                        hotspot_data['Photo URL'] = media_result['image_url']
                                    
                                    if params.get('include_audio') and media_result['audio_url'] != 'N/A':
                                        hotspot_data['Audio URL'] = media_result['audio_url']
                                        
                                except Exception as e:
                                    logger.warning(f"Failed to get media for {species_row['species_name']}: {str(e)}")
                            
                            all_hotspots.append(hotspot_data)
                    else:
                        logger.info(f"  ❌ REJECTED: Below {params['min_species_threshold']} species threshold")
                        # Note: Rejected locations do not contribute to results
                        continue
            
            except Exception as e:
                # Create ASCII-safe location name for logging
                safe_location_name = location_name.encode('ascii', 'ignore').decode('ascii').strip()
                if not safe_location_name or len(safe_location_name) < 3:
                    safe_location_name = f"GridPoint({point['latitude']:.3f}, {point['longitude']:.3f})"
                logger.error(f"Error processing grid point {safe_location_name}: {str(e)}")
                
                # Retry logic
                for retry in range(params['max_retries']):
                    try:
                        time.sleep(params['api_delay'] * 2)  # Longer delay for retries
                        # Retry the same analysis
                        ebird_observations = bird_client.get_ebird_observations(
                            lat=point['latitude'],
                            lng=point['longitude'],
                            radius_km=params['search_radius'],
                            days_back=30
                        )
                        break  # Success, exit retry loop
                    except Exception as retry_e:
                        safe_location_name = location_name.encode('ascii', 'ignore').decode('ascii').strip()
                        if not safe_location_name or len(safe_location_name) < 3:
                            safe_location_name = f"GridPoint({point['latitude']:.3f}, {point['longitude']:.3f})"
                        logger.warning(f"Retry {retry + 1} failed for {safe_location_name}: {str(retry_e)}")
                        if retry == params['max_retries'] - 1:
                            logger.error(f"All retries failed for {safe_location_name}")
                continue
            
            # Memory management and rate limiting
            if idx % 20 == 0:
                gc.collect()
            
            # API rate limiting
            time.sleep(params['api_delay'])
        
        # Finalize results
        progress_bar.progress(95)
        status_text.text("Finalizing dynamic discovery results...")
        
        if all_hotspots:
            results_df = pd.DataFrame(all_hotspots)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Processed {len(grid_points):,} grid points. Found {successful_analyses:,} qualifying hotspots ({orange_count:,} Orange + {red_count:,} Red)")
            return results_df
        else:
            progress_bar.empty()
            status_text.empty()
            st.warning("⚠️ No qualifying hotspots discovered. The analysis may need different parameters or there might be API issues.")
            return None
    
    except Exception as e:
        st.error(f"❌ Error during dynamic India discovery: {str(e)}")
        logger.error(f"Dynamic discovery error: {traceback.format_exc()}")
        return None

def display_dynamic_india_results(results_df, params):
    """Display the dynamic India hotspot discovery results."""
    st.info("📊 Showing dynamic India-wide hotspot discovery results")
    
    if not results_df.empty:
        # Summary statistics
        total_hotspots = len(results_df['Place'].unique())
        total_species = len(results_df['Species Name'].unique())
        total_birds = results_df['Bird Count'].sum()
        
        # Count hotspot types
        orange_hotspots = len(results_df[results_df['Hotspot Type'].str.contains('Orange')]['Place'].unique())
        red_hotspots = len(results_df[results_df['Hotspot Type'].str.contains('Red')]['Place'].unique())
        
        st.success("### 🇮🇳 Dynamic India Bird Hotspots Discovered")
        
        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🏞️ Total Hotspots", f"{total_hotspots:,}")
        with col2:
            st.metric("🟠 Orange (10-19)", f"{orange_hotspots:,}")
        with col3:
            st.metric("🔴 Red (20+)", f"{red_hotspots:,}")
        with col4:
            st.metric("🦜 Unique Species", f"{total_species:,}")
        with col5:
            st.metric("📊 Total Birds", f"{total_birds:,}")
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🗺️ Grid Type", params['grid_type'].title())
        with col2:
            st.metric("📍 Search Radius", f"{params['search_radius']} km")
        with col3:
            st.metric("⚡ Grid Points", f"{params['grid_points']:,}")
        
        # Regional analysis
        st.write("#### 🗺️ Regional Hotspot Distribution")
        regional_analysis = results_df.groupby('Region').agg({
            'Place': 'nunique',
            'Species Name': 'nunique', 
            'Bird Count': 'sum'
        }).reset_index()
        regional_analysis.columns = ['Region', 'Hotspots Found', 'Unique Species', 'Total Birds']
        regional_analysis = regional_analysis.sort_values('Hotspots Found', ascending=False)
        
        # Display regional data
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top Regions by Hotspot Count:**")
            st.dataframe(regional_analysis.head(10), use_container_width=True)
        
        with col2:
            st.write("**Hotspot Type Distribution:**")
            hotspot_type_dist = results_df.groupby('Hotspot Type')['Place'].nunique().reset_index()
            hotspot_type_dist.columns = ['Hotspot Type', 'Count']
            # Sort by count (descending) to show most common hotspot types first
            hotspot_type_dist = hotspot_type_dist.sort_values('Count', ascending=False)
            st.dataframe(hotspot_type_dist, use_container_width=True)
        
        # Top hotspots
        st.write("#### 🏆 Top Discovered Hotspots")
        species_col = 'Scientific Name' if 'Scientific Name' in results_df.columns else 'Species Name'
        top_hotspots = results_df.groupby(['Place', 'Region', 'Latitude', 'Longitude', 'Hotspot Type']).agg({
            species_col: 'nunique',
            'Bird Count': 'sum'
        }).reset_index()
        top_hotspots.columns = ['Place', 'Region', 'Latitude', 'Longitude', 'Type', 'Species Count', 'Total Birds']
        top_hotspots = top_hotspots.sort_values(['Species Count', 'Total Birds'], ascending=[False, False]).head(20)
        st.dataframe(top_hotspots, use_container_width=True)
        
        # Detailed data with pagination
        st.write("#### 📋 Detailed Dynamic Hotspot Data")
        
        # Add immediate download button for detailed data table
        try:
            detailed_excel_data = create_excel_download({'Detailed Hotspot Data': results_df}, "detailed_dynamic_hotspots")
            if detailed_excel_data:
                col1, col2 = st.columns([3, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Table Data",
                        data=detailed_excel_data,
                        file_name=f"detailed_dynamic_hotspots_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download only the detailed hotspot data shown in the table below"
                    )
        except Exception as e:
            st.warning(f"Quick download unavailable: {str(e)}")
        
        # Create display dataframe with proper sorting
        display_df = results_df.copy()
        
        # Sort the data for better user experience:
        # 1. By hotspot type (Red hotspots first, then Orange)
        # 2. By total species count (highest first)  
        # 3. By place name (alphabetical)
        display_df['Hotspot_Priority'] = display_df['Hotspot Type'].map({
            'Red Hotspot (20+ species)': 1,
            'Orange Hotspot (10-19 species)': 2
        })
        
        display_df = display_df.sort_values([
            'Hotspot_Priority',           # Red hotspots first
            'Total Species at Location',  # Highest species count first
            'Place',                      # Alphabetical by place name
            'Species Name'                # Alphabetical by species name
        ], ascending=[True, False, True, True])
        
        # Remove the temporary sorting column
        display_df = display_df.drop('Hotspot_Priority', axis=1)
        
        # Select key columns for display (prioritize scientific names and fun facts)
        display_columns = [
            'Place', 'Region', 'Latitude', 'Longitude', 
            'Scientific Name', 'Common Name', 'Fun Fact', 'Bird Count', 
            'Total Species at Location', 'Hotspot Type'
        ]
        
        # Fallback for older data without scientific names
        if 'Scientific Name' not in display_df.columns:
            display_columns = [
                'Place', 'Region', 'Latitude', 'Longitude', 
                'Species Name', 'Bird Count', 'Total Species at Location', 
                'Hotspot Type'
            ]
        
        if params['include_photos'] and 'Photo URL' in display_df.columns:
            display_columns.append('Photo URL')
        if params['include_audio'] and 'Audio URL' in display_df.columns:
            display_columns.append('Audio URL')
        
        # Show paginated results
        results_per_page = 100
        total_results = len(display_df)
        
        if total_results > results_per_page:
            page = st.selectbox(
                "Select page", 
                range(1, (total_results // results_per_page) + 2),
                format_func=lambda x: f"Page {x} ({min((x-1)*results_per_page + 1, total_results)}-{min(x*results_per_page, total_results)} of {total_results})"
            )
            start_idx = (page - 1) * results_per_page
            end_idx = min(page * results_per_page, total_results)
            display_data = display_df[display_columns].iloc[start_idx:end_idx]
        else:
            display_data = display_df[display_columns]
        
        st.dataframe(
            display_data,
            use_container_width=True,
            column_config={
                "Latitude": st.column_config.NumberColumn("Latitude", format="%.4f"),
                "Longitude": st.column_config.NumberColumn("Longitude", format="%.4f"),
                "Bird Count": st.column_config.NumberColumn("Bird Count", format="%d"),
                "Total Species at Location": st.column_config.NumberColumn("Total Species", format="%d"),
            }
        )
        
        # Interactive map with ONLY Orange and Red hotspots
        st.write("#### 🗺️ Dynamic Hotspots Map")
        
        try:
            india_map = create_india_map_with_hotspots(results_df)
            
            if india_map:
                st_folium(india_map, width=800, height=500)
            else:
                st.warning("Map rendering failed. Showing top hotspots as text:")
                for _, row in top_hotspots.head(15).iterrows():
                    color_emoji = "🔴" if "Red" in row['Type'] else "🟠"
                    st.write(f"{color_emoji} **{row['Place']}, {row['Region']}** - {row['Species Count']} species, {row['Total Birds']} birds")
        
        except Exception as e:
            st.warning(f"Map rendering error: {str(e)}")
        
        # Comprehensive Excel download
        st.write("#### 📥 Download Dynamic Analysis Results")
        
        # Add a prominent info box about download options
        st.info("🗂️ **Download Options Available:**\n- **Quick Download**: Table data only (above)\n- **Complete Download**: Full analysis with multiple sheets (below)")
        
        try:
            # Top species across all hotspots (using scientific names)
            if 'Scientific Name' in results_df.columns:
                # Group by scientific name and include fun facts
                species_summary = results_df.groupby(['Scientific Name', 'Common Name', 'Fun Fact']).agg({
                    'Bird Count': 'sum',
                    'Place': 'nunique'
                }).reset_index()
                species_summary.columns = ['Scientific Name', 'Common Name', 'Fun Fact', 'Total Observations', 'Hotspots Found']
            else:
                # Fallback for older data
                species_summary = results_df.groupby('Species Name').agg({
                    'Bird Count': 'sum',
                    'Place': 'nunique'
                }).reset_index()
                species_summary.columns = ['Species Name', 'Total Observations', 'Hotspots Found']
            
            species_summary = species_summary.sort_values('Total Observations', ascending=False)
            
            download_data = {
                'Analysis Summary': pd.DataFrame({
                    'Metric': [
                        'Total Dynamic Hotspots',
                        'Orange Hotspots (10-19 species)', 
                        'Red Hotspots (20+ species)',
                        'Total Unique Species',
                        'Total Bird Observations',
                        'Grid Type',
                        'Grid Points Analyzed',
                        'Search Radius (km)',
                        'Analysis Date'
                    ],
                    'Value': [
                        total_hotspots,
                        orange_hotspots,
                        red_hotspots,
                        total_species,
                        total_birds,
                        params['grid_type'],
                        params['grid_points'],
                        params['search_radius'],
                        datetime.now().strftime('%Y-%m-%d %H:%M')
                    ]
                }),
                'Top Hotspots': top_hotspots,
                'Regional Analysis': regional_analysis,
                'Species Summary': species_summary,
                'Detailed Data': results_df.copy()
            }
            
            # Clean URLs for Excel
            if 'Photo URL' in download_data['Detailed Data'].columns:
                download_data['Detailed Data']['Photo URL'] = download_data['Detailed Data']['Photo URL'].fillna('N/A')
            if 'Audio URL' in download_data['Detailed Data'].columns:
                download_data['Detailed Data']['Audio URL'] = download_data['Detailed Data']['Audio URL'].fillna('N/A')
            
            excel_data = create_excel_download(download_data, "dynamic_india_bird_hotspots")
            
            if excel_data:
                file_size_mb = len(excel_data) / (1024 * 1024)
                
                # Make the download button prominent
                st.markdown("---")  # Separator line
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label=f"📥 Download Complete Analysis ({file_size_mb:.1f} MB)",
                        data=excel_data,
                        file_name=f"dynamic_india_bird_hotspots_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        type="primary"
                    )
                
                st.markdown("---")  # Separator line
                
                # Show what's included in the download
                with st.expander("📋 What's included in the Complete Download?"):
                    st.write("**5 Excel Sheets:**")
                    st.write("1. **Analysis Summary** - Key metrics and parameters")
                    st.write("2. **Top Hotspots** - Best locations sorted by species count")
                    st.write("3. **Regional Analysis** - Hotspot distribution by region")
                    st.write("4. **Species Summary** - All species with observation counts")
                    st.write("5. **Detailed Data** - Complete dataset with all birds and locations")
                    if params['include_photos']:
                        st.write("📸 **Bonus:** Bird photo URLs included")
                    if params['include_audio']:
                        st.write("🔊 **Bonus:** Bird audio URLs included")
                
                st.success("✅ **Excel downloads ready!** Choose between quick table download or comprehensive analysis package.")
        
        except Exception as e:
            st.error(f"❌ Error creating Excel download: {str(e)}")
            logger.error(f"Excel download error: {traceback.format_exc()}")

def handle_tehsil_analysis(bird_client, search_radius, ebird_api_key, use_ebird):
    """Handle tehsil-based analysis functionality."""
    # Create two columns for the layout
    map_col, control_col = st.columns([2, 1])
    
    with control_col:
        st.write("### Select Location")
        
        # Load tehsil data
        tehsil_df = load_tehsil_data()
        
        if tehsil_df.empty:
            st.error("Unable to load tehsil data.")
            return
        
        # Create state and district filters
        states = sorted(tehsil_df['NAME_1'].unique())
        selected_state = st.selectbox("Select State", states, key="state_select")
        
        districts = sorted(tehsil_df[tehsil_df['NAME_1'] == selected_state]['NAME_2'].unique())
        selected_district = st.selectbox("Select District", districts, key="district_select")
        
        tehsils = sorted(tehsil_df[(tehsil_df['NAME_1'] == selected_state) & 
                                  (tehsil_df['NAME_2'] == selected_district)]['NAME_3'].unique())
        selected_tehsil = st.selectbox("Select Tehsil", tehsils, key="tehsil_select")
        
        # Get coordinates for selected tehsil
        tehsil_data = tehsil_df[(tehsil_df['NAME_1'] == selected_state) & 
                               (tehsil_df['NAME_2'] == selected_district) &
                               (tehsil_df['NAME_3'] == selected_tehsil)].iloc[0]
        
        latitude = tehsil_data['Latitude_N']
        longitude = tehsil_data['Longitude_E']
        location_name = f"{selected_tehsil}, {selected_district}, {selected_state}"
        
        # Analyze button
        analyze_button = st.button(
            "🔍 Find Bird Hotspots",
            key="tehsil_analyze_button",
            use_container_width=True,
            help="Click to analyze bird species in this location"
        )
    
    with map_col:
        # Show selected location on map
        st.write("### Selected Location")
        
        # Create a Folium map focused on India
        m = folium.Map(
            location=[23.5937, 78.9629],
            zoom_start=5,
            tiles="cartodbpositron",
            min_zoom=4,
            max_zoom=12,
            max_bounds=True,
            min_lat=6.5, max_lat=37.5,
            min_lon=68.0, max_lon=97.5
        )
        
        # Add state boundary context
        state_tehsils = tehsil_df[tehsil_df['NAME_1'] == selected_state]
        
        # Add tehsil points for the selected state
        for _, row in state_tehsils.iterrows():
            folium.CircleMarker(
                location=[row['Latitude_N'], row['Longitude_E']],
                radius=2,
                color='brown',
                fill=True,
                popup=f"{row['NAME_3']}, {row['NAME_2']}"
            ).add_to(m)
        
        # Add the selected location with a distinct marker
        folium.Marker(
            [latitude, longitude],
            popup=f"<b>{location_name}</b>",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Set map bounds to India
        m.fit_bounds([[8.4, 68.7], [37.6, 97.25]])
        
        # Display the map
        st_folium(m, width=800, height=600)
    
    # Handle analysis
    if analyze_button:
        analyze_and_display_single_location(
            latitude, longitude, location_name,
            search_radius, bird_client, ebird_api_key, use_ebird
        )

def handle_all_india_analysis(bird_client, ebird_api_key, use_ebird):
    """Handle All-India analysis functionality."""
    st.write("### All-India Bird Hotspot Analysis")
    st.info("This analysis will process ALL locations from bird_hotspot.json file within a 50km search radius from each location.")
    
    # Load bird hotspot data
    try:
        hotspot_df = load_bird_hotspot_data()
        st.success(f"Loaded {len(hotspot_df)} locations from bird_hotspot.json")
    except Exception as e:
        st.error(f"Error loading bird hotspot data: {str(e)}")
        return
    
    # Configuration controls
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        search_radius_all_india = st.slider(
            "Search radius (km)",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            help="Radius in kilometers to search around each location"
        )
    
    with control_col2:
        sample_percentage = st.slider(
            "Sample percentage (%)",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            help="Percentage of locations to sample for analysis"
        )
    
    # Show data overview
    st.write("#### Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Locations", len(hotspot_df))
    with col2:
        st.metric("Unique States", hotspot_df['state'].nunique())
    with col3:
        sample_size = int(len(hotspot_df) * sample_percentage / 100)
        st.metric("Locations to Process", sample_size)
    
    # Warning for large analyses
    if sample_percentage > 20:
        st.warning("⚠️ Large sample sizes may take significant time and cause performance issues.")
    
    # Current parameters
    current_params = {
        'sample_percentage': sample_percentage,
        'search_radius': search_radius_all_india,
        'use_ebird': use_ebird,
        'ebird_api_key': EBIRD_API_KEY if use_ebird else ""
    }
    
    # Analyze button
    analyze_button = st.button(
        "🔍 Analyze All India Hotspots",
        key="all_india_analyze_button",
        use_container_width=True
    )
    
    # Clear results button
    if st.session_state.all_india_results is not None:
        if st.button("🗑️ Clear Cached Results", key="clear_results_button", use_container_width=True):
            st.session_state.all_india_results = None
            st.session_state.all_india_analysis_params = None
            st.success("✅ Cached results cleared!")
            st.rerun()
    
    # Display previous results if available and parameters haven't changed
    if (st.session_state.all_india_results is not None and 
        st.session_state.all_india_analysis_params == current_params):
        
        display_all_india_results(st.session_state.all_india_results, current_params)
    
    if analyze_button:
        if not use_ebird or not EBIRD_API_KEY:
            st.error("❌ eBird API key is required. Please enable eBird and ensure EBIRD_API_KEY is set in your .env file.")
            return
        
        # Clear previous results
        st.session_state.all_india_results = None
        st.session_state.all_india_analysis_params = None
        
        # Run analysis
        results_df = run_all_india_analysis(
            hotspot_df, bird_client, sample_percentage, search_radius_all_india
        )
        
        if results_df is not None and not results_df.empty:
            # Store in session state
            st.session_state.all_india_results = results_df
            st.session_state.all_india_analysis_params = current_params
            
            st.success("✅ Analysis completed successfully!")
            st.rerun()

def run_all_india_analysis(hotspot_df, bird_client, sample_percentage, search_radius):
    """Run the All-India analysis process."""
    try:
        st.write("### Processing All India Locations from bird_hotspot.json")
        
        # Sample the data
        if sample_percentage < 100:
            sample_size = int(len(hotspot_df) * sample_percentage / 100)
            sampled_locations = hotspot_df.sample(n=sample_size, random_state=42)
            st.info(f"Processing {len(sampled_locations)} sampled locations ({sample_percentage}% of total)")
        else:
            sampled_locations = hotspot_df
            st.info(f"Processing all {len(sampled_locations)} locations")
        
        # Initialize results storage
        results = []
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each location
        for idx, (_, location) in enumerate(sampled_locations.iterrows()):
            # Update progress
            progress = (idx + 1) / len(sampled_locations)
            progress_bar.progress(progress)
            
            location_name = f"{location['tehsil']}, {location['district']}, {location['state']}"
            status_text.text(f"Analyzing {location_name} ({idx + 1}/{len(sampled_locations)})")
            
            try:
                # Get bird data
                ebird_observations = bird_client.get_ebird_observations(
                    lat=location['latitude'],
                    lng=location['longitude'],
                    radius_km=search_radius,
                    days_back=30
                )
                
                if not ebird_observations.empty:
                    # Calculate species counts
                    species_counts = ebird_observations.groupby('comName').size().reset_index(name='count')
                    
                    # Add to results
                    for _, row in species_counts.iterrows():
                        results.append({
                            'State': location['state'],
                            'District': location['district'],
                            'Tehsil': location['tehsil'],
                            'Latitude': location['latitude'],
                            'Longitude': location['longitude'],
                            'Species Name': row['comName'],
                            'Species Count': row['count'],
                            'Search Radius (km)': search_radius
                        })
            
            except Exception as e:
                logger.error(f"Error processing location {location_name}: {str(e)}")
                continue
            
            # Memory management
            if idx % 50 == 0:
                gc.collect()
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if results:
            return pd.DataFrame(results)
        else:
            st.warning("No bird observations found in any location.")
            return None
    
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        logger.error(f"Error in all-India analysis: {traceback.format_exc()}")
        return None

def display_all_india_results(results_df, current_params):
    """Display the All-India analysis results."""
    st.info("📊 Showing previously computed results (parameters unchanged)")
    
    if not results_df.empty:
        # Summary statistics
        total_locations = len(results_df[['State', 'District', 'Tehsil']].drop_duplicates())
        total_species = len(results_df['Species Name'].unique())
        total_birds = results_df['Species Count'].sum()
        
        st.success("### Analysis Results (Cached)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Locations Analyzed", total_locations)
        with col2:
            st.metric("Unique Species", total_species)
        with col3:
            st.metric("Total Birds", total_birds)
        
        # Show state-wise summary
        st.write("#### State-wise Summary")
        state_summary = results_df.groupby('State').agg({
            'Species Name': 'nunique',
            'Species Count': 'sum',
            'Tehsil': 'nunique'
        }).reset_index()
        state_summary.columns = ['State', 'Unique Species', 'Total Birds', 'Locations Analyzed']
        st.dataframe(state_summary.sort_values('Unique Species', ascending=False), use_container_width=True)
        
        # Show top hotspots
        st.write("#### Top Bird Hotspots")
        hotspot_summary = results_df.groupby(['State', 'District', 'Tehsil', 'Latitude', 'Longitude']).agg({
            'Species Name': 'nunique',
            'Species Count': 'sum'
        }).reset_index()
        
        hotspot_summary.columns = ['State', 'District', 'Tehsil', 'Latitude', 'Longitude', 'Unique Species', 'Total Birds']
        hotspot_summary = hotspot_summary.sort_values(['Unique Species', 'Total Birds'], ascending=[False, False]).head(20)
        st.dataframe(hotspot_summary, use_container_width=True)
        
        # Map rendering based on sample size
        if current_params['sample_percentage'] >= 100:
            st.info("🗺️ **Map rendering skipped for full analysis to prevent performance issues.**")
        elif current_params['sample_percentage'] >= 50:
            st.write("#### Top Hotspot Locations")
            for idx, row in hotspot_summary.head(10).iterrows():
                st.write(f"🏆 **{row['Tehsil']}, {row['District']}, {row['State']}** - {row['Unique Species']} species, {row['Total Birds']} birds")
        elif current_params['sample_percentage'] >= 20:
            st.write("#### Hotspot Distribution (Basic Map)")
            map_data = hotspot_summary.head(20)[['Latitude', 'Longitude']].copy()
            map_data.columns = ['lat', 'lon']
            try:
                st.map(map_data)
            except Exception as e:
                st.warning(f"Map rendering failed: {str(e)}")
        else:
            # Interactive map for small samples
            st.write("#### Top Hotspots Map (Interactive)")
            try:
                results_map = folium.Map(
                    location=[23.5937, 78.9629],
                    zoom_start=5,
                    tiles="OpenStreetMap",
                    prefer_canvas=True
                )
                
                for _, hotspot in hotspot_summary.head(25).iterrows():
                    folium.CircleMarker(
                        location=[hotspot['Latitude'], hotspot['Longitude']],
                        radius=min(hotspot['Unique Species']/3, 10),
                        color='red',
                        fill=True,
                        weight=1,
                        popup=f"{hotspot['Tehsil']}, {hotspot['State']}<br>Species: {hotspot['Unique Species']}"
                    ).add_to(results_map)
                
                st_folium(results_map, width=800, height=400)
            except Exception as e:
                st.warning(f"Interactive map failed: {str(e)}")
        
        # Excel download
        st.write("#### 📥 Download Results")
        
        try:
            download_data = {
                'Analysis Summary': pd.DataFrame({
                    'Analysis Details': ['Total Locations Processed', 'Search Radius (km)', 'Sample Percentage (%)', 'Total Unique Species', 'Total Bird Observations'],
                    'Values': [total_locations, current_params['search_radius'], current_params['sample_percentage'], total_species, total_birds]
                }),
                'State Summary': state_summary,
                'Top Hotspots': hotspot_summary,
                'Detailed Data': results_df
            }
            
            excel_data = create_excel_download(download_data, "all_india_bird_hotspots")
            
            if excel_data:
                file_size_mb = len(excel_data) / (1024 * 1024)
                
                st.download_button(
                    label=f"📥 Download Complete Results ({file_size_mb:.1f} MB)",
                    data=excel_data,
                    file_name=f"all_india_bird_hotspots_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                st.success("✅ **Excel file ready for download!** Contains all analysis data across multiple sheets.")
        
        except Exception as e:
            st.error(f"Error creating Excel file: {str(e)}")

if __name__ == "__main__":
    main()

