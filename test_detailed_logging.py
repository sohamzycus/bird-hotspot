#!/usr/bin/env python3
"""
Test script to demonstrate enhanced logging with sample locations
"""

import os
import logging
from dotenv import load_dotenv
from bird_api_client import BirdDataClient
from bird_hotspot_ui import check_habitat_viability

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detailed_test.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("detailed_test")

def test_detailed_search_logging():
    """Test the enhanced search logging with various types of locations"""
    print("🔍 Testing Enhanced Search Logging")
    print("=" * 50)
    
    # Get API key
    ebird_api_key = os.getenv("EBIRD_API_KEY", "")
    if not ebird_api_key:
        print("❌ No eBird API key found")
        return
    
    # Initialize client
    client = BirdDataClient(ebird_api_key=ebird_api_key)
    
    # Test locations with different characteristics
    test_locations = [
        {"name": "Delhi (Good Urban)", "lat": 28.6139, "lng": 77.2090},
        {"name": "Mumbai Coast (Good Coastal)", "lat": 19.0760, "lng": 72.8777},
        {"name": "Arabian Sea (Ocean)", "lat": 18.0, "lng": 68.0},
        {"name": "Bay of Bengal (Ocean)", "lat": 16.0, "lng": 93.0},
        {"name": "Thar Desert (Arid)", "lat": 27.0, "lng": 71.0},
        {"name": "High Himalayas (Sparse)", "lat": 34.0, "lng": 78.0}
    ]
    
    search_radius = 25
    
    for location in test_locations:
        print(f"\n📍 Testing: {location['name']}")
        
        # Check habitat viability
        habitat = check_habitat_viability(location['lat'], location['lng'])
        print(f"   🌍 Habitat Assessment: {habitat}")
        
        logger.info(f"🔍 SEARCHING: {location['name']} | Coords: ({location['lat']:.4f}, {location['lng']:.4f}) | Radius: {search_radius}km | Habitat: {habitat}")
        
        # Test eBird search
        try:
            ebird_data = client.get_ebird_observations(
                lat=location['lat'],
                lng=location['lng'],
                radius_km=search_radius,
                days_back=30
            )
            
            if not ebird_data.empty:
                species_count = ebird_data['comName'].nunique()
                print(f"   🐦 eBird: {len(ebird_data)} observations, {species_count} species")
                logger.info(f"  📊 eBird: {len(ebird_data)} observations → {species_count} species")
                
                # Show sample species
                if species_count > 0:
                    sample_species = ebird_data['comName'].unique()[:3]
                    print(f"   📋 Sample species: {', '.join(sample_species)}")
                    logger.info(f"  🐦 Sample eBird species: {', '.join(sample_species)}")
            else:
                print(f"   ❌ eBird: No data found")
                logger.warning(f"  ❌ eBird: No observations found in {search_radius}km radius")
                
        except Exception as e:
            print(f"   ⚠️ eBird error: {str(e)}")
            logger.error(f"  ⚠️ eBird API error: {str(e)}")
        
        # Test GBIF search
        try:
            gbif_data = client.get_gbif_occurrences(
                lat=location['lat'],
                lng=location['lng'],
                radius_km=search_radius
            )
            
            if not gbif_data.empty and 'species' in gbif_data.columns:
                species_count = gbif_data['species'].nunique()
                print(f"   🔬 GBIF: {len(gbif_data)} occurrences, {species_count} species")
                logger.info(f"  📊 GBIF: {len(gbif_data)} occurrences → {species_count} species")
                
                # Show sample species
                if species_count > 0:
                    sample_species = gbif_data['species'].unique()[:3]
                    print(f"   📋 Sample GBIF species: {', '.join(sample_species)}")
                    logger.info(f"  🔬 Sample GBIF species: {', '.join(sample_species)}")
            else:
                print(f"   ❌ GBIF: No data found")
                logger.warning(f"  ❌ GBIF: No valid occurrences found in {search_radius}km radius")
                
        except Exception as e:
            print(f"   ⚠️ GBIF error: {str(e)}")
            logger.error(f"  ⚠️ GBIF API error: {str(e)}")

if __name__ == "__main__":
    test_detailed_search_logging()
    print(f"\n📄 Detailed logs saved to: detailed_test.log")
    print("🎯 This demonstrates what you'll see in the enhanced UI logging!") 