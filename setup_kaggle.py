#!/usr/bin/env python3
"""
Kaggle API Setup Helper for Perch 2.0 Integration
Helps users configure Kaggle API credentials
"""

import os
import json
from pathlib import Path

def setup_kaggle_api():
    """Interactive setup for Kaggle API credentials."""
    print("🎵 Perch 2.0 - Kaggle API Setup Helper")
    print("=" * 50)
    
    # Get user's home directory
    home_dir = Path.home()
    kaggle_dir = home_dir / ".kaggle"
    kaggle_json_path = kaggle_dir / "kaggle.json"
    
    print(f"📂 Kaggle directory: {kaggle_dir}")
    print(f"📄 Config file: {kaggle_json_path}")
    print()
    
    # Check if already exists
    if kaggle_json_path.exists():
        print("✅ kaggle.json already exists!")
        with open(kaggle_json_path, 'r') as f:
            config = json.load(f)
            username = config.get('username', 'Unknown')
            print(f"   Username: {username}")
        
        choice = input("\n🔄 Do you want to update credentials? (y/n): ").lower()
        if choice != 'y':
            print("✅ Kaggle API already configured!")
            return True
    
    print("\n📋 **Steps to get Kaggle API credentials:**")
    print("1. Go to https://www.kaggle.com")
    print("2. Create account or login")
    print("3. Go to Account → API → Create New API Token")
    print("4. Download kaggle.json file")
    print("5. Enter the credentials below")
    print()
    
    # Get credentials from user
    username = input("Enter your Kaggle username: ").strip()
    if not username:
        print("❌ Username cannot be empty!")
        return False
    
    api_key = input("Enter your Kaggle API key: ").strip()
    if not api_key:
        print("❌ API key cannot be empty!")
        return False
    
    # Create kaggle directory if it doesn't exist
    kaggle_dir.mkdir(exist_ok=True)
    
    # Create kaggle.json
    config = {
        "username": username,
        "key": api_key
    }
    
    try:
        with open(kaggle_json_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set appropriate permissions (600 = read/write for owner only)
        os.chmod(kaggle_json_path, 0o600)
        
        print(f"\n✅ Successfully created {kaggle_json_path}")
        print("✅ Permissions set to 600 (secure)")
        print("\n🎵 **Ready to download Perch 2.0 model!**")
        print("   Run: streamlit run bird_hotspot_ui.py")
        return True
        
    except Exception as e:
        print(f"❌ Error creating kaggle.json: {e}")
        return False

def test_kaggle_connection():
    """Test if Kaggle API is working."""
    try:
        import kaggle
        print("✅ Kaggle API authentication successful!")
        
        # Try to list datasets (minimal API call)
        print("🔍 Testing API connection...")
        kaggle.api.dataset_list(search="bird", page_size=1)
        print("✅ Kaggle API connection working!")
        return True
        
    except ImportError:
        print("❌ Kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Kaggle API error: {e}")
        return False

if __name__ == "__main__":
    print("🎵 Perch 2.0 Kaggle Setup")
    print("=" * 30)
    
    # Setup credentials
    if setup_kaggle_api():
        print("\n" + "=" * 30)
        print("🧪 Testing connection...")
        test_kaggle_connection()
    else:
        print("\n❌ Setup failed. Please try again.")
    
    print("\n" + "=" * 30)
    print("📖 **Next Steps:**")
    print("1. Run: streamlit run bird_hotspot_ui.py")
    print("2. Navigate to '🎵 Perch 2.0 Bird Sound Testing'")
    print("3. Click 'Download & Load Perch 2.0 Model'")
    print("4. Start testing bird sounds!")
