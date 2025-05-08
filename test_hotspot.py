# test_hotspot.py
import os
import sys
import traceback
from datetime import datetime
from south_asian_bird_hotspot import SouthAsianBirdHotspotPredictor

def test_hotspot_algorithm():
    print("Testing South Asian Bird Hotspot Algorithm")
    
    try:
        # Define a small region for testing (Delhi area)
        region_bbox = (28.5, 77.0, 28.7, 77.2)
        
        # Initialize predictor
        print("Initializing predictor...")
        predictor = SouthAsianBirdHotspotPredictor(region_bbox, grid_size=0.02)
        
        # Create grid
        print("Creating grid...")
        grid = predictor.create_grid()
        print(f"Grid created with {len(grid)} points")
        
        # Extract environmental features
        print("Extracting environmental features...")
        grid = predictor.extract_south_asian_environmental_features(grid)
        print("Environmental features extracted")
        
        # Identify key habitats
        print("Identifying key habitats...")
        grid = predictor.identify_key_habitats(grid)
        print("Key habitats identified")
        
        # Process for current date
        print("Processing for current date...")
        result = predictor.process_for_current_date()
        
        # Check results
        if result and 'grid' in result and len(result['grid']) > 0:
            print(f"SUCCESS: Algorithm produced valid results with {len(result['grid'])} grid points")
            
            # Save results for inspection
            os.makedirs("test_output", exist_ok=True)
            result['grid'].drop('geometry', axis=1).to_csv("test_output/test_grid.csv", index=False)
            
            if 'hotspots' in result and result['hotspots'] is not None:
                print(f"Found {len(result['hotspots'])} hotspots")
                result['hotspots'].drop('geometry', axis=1).to_csv("test_output/test_hotspots.csv", index=False)
            
            print("Test results saved to test_output directory")
            return True
        else:
            print("ERROR: Algorithm did not produce valid results")
            return False
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_hotspot_algorithm()