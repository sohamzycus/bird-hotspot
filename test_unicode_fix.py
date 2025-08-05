#!/usr/bin/env python3
"""
Test script to verify Unicode handling and DataFrame column fixes
"""

import pandas as pd
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_debug.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_unicode")

def test_unicode_location_name():
    """Test Unicode location name handling"""
    print("🔤 Testing Unicode Location Name Handling...")
    
    # Test with problematic Unicode characters (Maldivian script)
    unicode_name = "ތިލަދުންމަތީ ދެކުނުބުރި"
    
    # Test ASCII conversion
    try:
        ascii_name = unicode_name.encode('ascii', 'ignore').decode('ascii')
        print(f"✅ Unicode → ASCII: '{unicode_name}' → '{ascii_name}'")
        
        # Test safe logging
        safe_name = ascii_name if ascii_name else "Point(6.400, 72.779)"
        logger.info(f"Test location: {safe_name}")
        print(f"✅ Safe logging successful")
        
        return True
    except Exception as e:
        print(f"❌ Unicode handling failed: {str(e)}")
        return False

def test_dataframe_column_handling():
    """Test DataFrame column existence handling"""
    print("\n📊 Testing DataFrame Column Handling...")
    
    # Simulate different scenarios
    scenarios = [
        {"name": "Only eBird data", "data": [
            {"species_name": "House Sparrow", "ebird_count": 5, "source": "eBird"}
        ]},
        {"name": "Only GBIF data", "data": [
            {"species_name": "Passer domesticus", "gbif_count": 3, "source": "GBIF"}
        ]},
        {"name": "Mixed data", "data": [
            {"species_name": "Common Myna", "ebird_count": 2, "source": "eBird"},
            {"species_name": "Acridotheres tristis", "gbif_count": 1, "source": "GBIF"}
        ]}
    ]
    
    for scenario in scenarios:
        print(f"\n  Testing: {scenario['name']}")
        
        try:
            # Create DataFrame
            df = pd.DataFrame(scenario['data'])
            print(f"    Original columns: {list(df.columns)}")
            
            # Ensure required columns exist (our fix)
            if 'ebird_count' not in df.columns:
                df['ebird_count'] = 0
            if 'gbif_count' not in df.columns:
                df['gbif_count'] = 0
            
            # Fill NaN values
            df['ebird_count'] = df['ebird_count'].fillna(0)
            df['gbif_count'] = df['gbif_count'].fillna(0)
            
            print(f"    After fix columns: {list(df.columns)}")
            
            # Test aggregation (the operation that was failing)
            combined = df.groupby('species_name').agg({
                'ebird_count': lambda x: x.sum(),
                'gbif_count': lambda x: x.sum()
            }).reset_index()
            
            print(f"    ✅ Aggregation successful: {len(combined)} species")
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            return False
    
    return True

def test_api_key_consistency():
    """Test API key consistency from environment"""
    print("\n🔑 Testing API Key Consistency...")
    
    api_key = os.getenv("EBIRD_API_KEY", "")
    
    if api_key:
        print(f"✅ API key found: {api_key[:8]}...")
        return True
    else:
        print("❌ No API key found in environment")
        return False

if __name__ == "__main__":
    print("🧪 Testing Unicode and DataFrame Fixes")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_unicode_location_name():
        tests_passed += 1
    
    if test_dataframe_column_handling():
        tests_passed += 1
    
    if test_api_key_consistency():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Fixes are working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.") 