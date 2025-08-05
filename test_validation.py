#!/usr/bin/env python3
"""
Comprehensive validation test for bird hotspot analysis
Tests grid generation, data processing, and hotspot calculations
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import our modules
from bird_api_client import BirdDataClient
from bird_hotspot_ui import generate_dynamic_india_grid, get_real_location_name, get_dynamic_region_name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("validation")

def test_grid_generation():
    """Test grid generation for all three types"""
    print("🔍 Testing Grid Generation...")
    
    # Test systematic grid
    systematic_grid = generate_dynamic_india_grid(max_points=10, grid_type="systematic")
    print(f"✅ Systematic grid: {len(systematic_grid)} points")
    
    # Test adaptive grid  
    adaptive_grid = generate_dynamic_india_grid(max_points=10, grid_type="adaptive")
    print(f"✅ Adaptive grid: {len(adaptive_grid)} points")
    
    # Test dense grid
    dense_grid = generate_dynamic_india_grid(max_points=10, grid_type="dense")
    print(f"✅ Dense grid: {len(dense_grid)} points")
    
    # Validate grid structure
    for grid_type, grid in [("systematic", systematic_grid), ("adaptive", adaptive_grid), ("dense", dense_grid)]:
        if grid.empty:
            print(f"❌ {grid_type} grid is empty!")
            continue
            
        required_cols = ['id', 'latitude', 'longitude', 'location_name', 'grid_type']
        missing_cols = [col for col in required_cols if col not in grid.columns]
        if missing_cols:
            print(f"❌ {grid_type} grid missing columns: {missing_cols}")
        else:
            print(f"✅ {grid_type} grid has all required columns")
        
        # Check coordinates are within India bounds
        lat_range = (6.4, 37.6)
        lng_range = (68.7, 97.25)
        
        invalid_lats = grid[(grid['latitude'] < lat_range[0]) | (grid['latitude'] > lat_range[1])]
        invalid_lngs = grid[(grid['longitude'] < lng_range[0]) | (grid['longitude'] > lng_range[1])]
        
        if len(invalid_lats) > 0:
            print(f"❌ {grid_type} grid has {len(invalid_lats)} points with invalid latitudes")
        if len(invalid_lngs) > 0:
            print(f"❌ {grid_type} grid has {len(invalid_lngs)} points with invalid longitudes")
        
        # Check for unique location names
        unique_names = grid['location_name'].nunique()
        total_points = len(grid)
        if unique_names < total_points:
            print(f"⚠️ {grid_type} grid has duplicate location names: {unique_names}/{total_points} unique")
        else:
            print(f"✅ {grid_type} grid has unique location names")
        
        # Show sample location names
        print(f"📍 Sample {grid_type} locations:")
        for i, row in grid.head(3).iterrows():
            print(f"   - {row['location_name']}")
    
    return systematic_grid, adaptive_grid, dense_grid

def test_location_naming():
    """Test location naming functions"""
    print("\n🏷️ Testing Location Naming...")
    
    # Test coordinates across India
    test_coords = [
        (28.6139, 77.2090),  # Delhi
        (19.0760, 72.8777),  # Mumbai
        (13.0827, 80.2707),  # Chennai
        (22.5726, 88.3639),  # Kolkata
        (12.9716, 77.5946),  # Bangalore
    ]
    
    for lat, lng in test_coords:
        region_name = get_dynamic_region_name(lat, lng)
        print(f"✅ ({lat:.3f}, {lng:.3f}) → {region_name}")
    
    print("✅ Location naming test completed")

def test_data_processing_logic():
    """Test the data combination and species counting logic"""
    print("\n🧮 Testing Data Processing Logic...")
    
    # Create mock eBird data
    ebird_data = pd.DataFrame({
        'comName': ['House Sparrow', 'Common Myna', 'Indian Robin', 'House Sparrow'],
        'lat': [28.6, 28.6, 28.6, 28.6],
        'lng': [77.2, 77.2, 77.2, 77.2]
    })
    
    # Create mock GBIF data  
    gbif_data = pd.DataFrame({
        'species': ['Passer domesticus', 'Acridotheres tristis', 'Copsychus fulicatus', 'Corvus splendens'],
        'decimalLatitude': [28.6, 28.6, 28.6, 28.6],
        'decimalLongitude': [77.2, 77.2, 77.2, 77.2]
    })
    
    # Test eBird processing
    if not ebird_data.empty:
        ebird_species = ebird_data.groupby('comName').size().reset_index(name='ebird_count')
        ebird_species['source'] = 'eBird'
        ebird_species.rename(columns={'comName': 'species_name'}, inplace=True)
        print(f"✅ eBird data: {len(ebird_species)} unique species")
    
    # Test GBIF processing
    if not gbif_data.empty:
        gbif_species = gbif_data.groupby('species').size().reset_index(name='gbif_count')
        gbif_species['source'] = 'GBIF'
        gbif_species.rename(columns={'species': 'species_name'}, inplace=True)
        print(f"✅ GBIF data: {len(gbif_species)} unique species")
    
    # Test combined processing
    all_species = pd.concat([ebird_species, gbif_species], ignore_index=True)
    
    # Combine species from both sources properly
    combined_species = all_species.groupby('species_name').agg({
        'ebird_count': lambda x: x.fillna(0).sum(),
        'gbif_count': lambda x: x.fillna(0).sum()
    }).reset_index()
    
    # Fill missing columns with 0
    if 'ebird_count' not in combined_species.columns:
        combined_species['ebird_count'] = 0
    if 'gbif_count' not in combined_species.columns:
        combined_species['gbif_count'] = 0
        
    # Fill NaN values with 0
    combined_species['ebird_count'] = combined_species['ebird_count'].fillna(0)
    combined_species['gbif_count'] = combined_species['gbif_count'].fillna(0)
    
    # Calculate combined count
    combined_species['combined_count'] = combined_species['ebird_count'] + combined_species['gbif_count']
    
    unique_species_count = len(combined_species)
    total_bird_count = int(combined_species['combined_count'].sum())
    
    print(f"✅ Combined data: {unique_species_count} unique species, {total_bird_count} total observations")
    print(f"📊 Combined species breakdown:")
    for _, row in combined_species.iterrows():
        source = 'eBird' if row['ebird_count'] > 0 else 'GBIF'
        if row['ebird_count'] > 0 and row['gbif_count'] > 0:
            source = 'eBird+GBIF'
        print(f"   - {row['species_name']}: {int(row['combined_count'])} ({source})")
    
    # Test hotspot classification
    if unique_species_count >= 20:
        hotspot_type = "Red Hotspot (20+ species)"
    elif unique_species_count >= 10:
        hotspot_type = "Orange Hotspot (10-19 species)"
    else:
        hotspot_type = "Below threshold (< 10 species)"
    
    print(f"🎯 Hotspot classification: {hotspot_type}")
    
    return combined_species

def test_api_client():
    """Test API client functionality"""
    print("\n🌐 Testing API Client...")
    
    # Get API key from environment
    import os
    from dotenv import load_dotenv
    load_dotenv()
    ebird_api_key = os.getenv("EBIRD_API_KEY", "")
    
    # Initialize client with API key
    client = BirdDataClient(ebird_api_key=ebird_api_key)
    
    # Test with coordinates near Delhi (should have data)
    test_lat, test_lng = 28.6139, 77.2090
    
    print(f"Testing APIs for Delhi area ({test_lat}, {test_lng})...")
    
    # Test eBird (without API key, should handle gracefully)
    try:
        ebird_data = client.get_ebird_observations(test_lat, test_lng, radius_km=25)
        if not ebird_data.empty:
            print(f"✅ eBird: {len(ebird_data)} observations, {ebird_data['comName'].nunique()} species")
        else:
            print("⚠️ eBird: No data (API key may be required)")
    except Exception as e:
        print(f"⚠️ eBird error: {str(e)}")
    
    # Test GBIF
    try:
        gbif_data = client.get_gbif_occurrences(test_lat, test_lng, radius_km=25)
        if not gbif_data.empty:
            if 'species' in gbif_data.columns:
                print(f"✅ GBIF: {len(gbif_data)} occurrences, {gbif_data['species'].nunique()} species")
            else:
                print(f"⚠️ GBIF: {len(gbif_data)} occurrences, but no 'species' column")
                print(f"   Available columns: {list(gbif_data.columns)}")
        else:
            print("⚠️ GBIF: No data returned")
    except Exception as e:
        print(f"❌ GBIF error: {str(e)}")

def run_comprehensive_validation():
    """Run all validation tests"""
    print("🔬 COMPREHENSIVE BIRD HOTSPOT VALIDATION")
    print("=" * 50)
    
    # Test 1: Grid generation
    grids = test_grid_generation()
    
    # Test 2: Location naming
    test_location_naming()
    
    # Test 3: Data processing
    test_data_processing_logic()
    
    # Test 4: API client
    test_api_client()
    
    print("\n" + "=" * 50)
    print("✅ VALIDATION COMPLETE!")
    print("Check the results above for any issues that need fixing.")

if __name__ == "__main__":
    run_comprehensive_validation() 