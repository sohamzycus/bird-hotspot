#!/usr/bin/env python3
"""
Simple test to validate eBird API key from environment
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_ebird_api_key():
    """Test if the eBird API key from .env file is working"""
    
    # Get API key from environment
    api_key = os.getenv("EBIRD_API_KEY", "")
    
    if not api_key:
        print("❌ No eBird API key found in .env file")
        print("💡 Add EBIRD_API_KEY=your_key_here to your .env file")
        return False
    
    print(f"✅ Found eBird API key: {api_key[:8]}...")
    
    # Test API call to a known location (Delhi, India)
    url = "https://api.ebird.org/v2/data/obs/geo/recent"
    params = {
        "lat": 28.6139,
        "lng": 77.2090,
        "dist": 25,
        "back": 7,
        "fmt": "json"
    }
    headers = {
        "X-eBirdApiToken": api_key
    }
    
    try:
        print("🔍 Testing API call to eBird...")
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API call successful! Found {len(data)} bird observations")
            if data:
                print(f"📋 Sample species: {data[0].get('comName', 'Unknown')}")
            return True
        elif response.status_code == 403:
            print("❌ API returned 403 Forbidden - Invalid API key or no access")
            print("🔗 Get a valid API key from: https://ebird.org/api/keygen")
            return False
        elif response.status_code == 429:
            print("⚠️ API rate limit exceeded - try again later")
            return False
        else:
            print(f"❌ API call failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔬 Testing eBird API Key Configuration")
    print("=" * 50)
    
    success = test_ebird_api_key()
    
    if success:
        print("\n🎉 eBird API key is working correctly!")
    else:
        print("\n💡 To get a valid eBird API key:")
        print("   1. Visit: https://ebird.org/api/keygen")
        print("   2. Sign in with your eBird account")
        print("   3. Request an API key")
        print("   4. Add it to your .env file as: EBIRD_API_KEY=your_key_here") 