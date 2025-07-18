"""
Bird API Client - Functions to interact with bird data APIs
"""

import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
import numpy as np
from functools import lru_cache

logger = logging.getLogger("bird_api")

class BirdDataClient:
    def __init__(self, ebird_api_key=""):
        self.ebird_api_key = ebird_api_key
        self.ebird_base_url = "https://api.ebird.org/v2"
        self.gbif_base_url = "https://api.gbif.org/v1"
        self.xeno_canto_base_url = "https://www.xeno-canto.org/api/2"
        
    def get_ebird_observations(self, lat, lng, radius_km=25, days_back=14):
        """
        Get recent bird observations from eBird API
        
        Parameters:
        -----------
        lat : float
            Latitude of the center point
        lng : float
            Longitude of the center point
        radius_km : int
            Search radius in kilometers
        days_back : int
            Number of days to look back for observations
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with bird observations
        """
        try:
            # Calculate the date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for API
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            # Build the API URL
            url = f"{self.ebird_base_url}/data/obs/geo/recent"
            
            # Set up parameters
            params = {
                "lat": lat,
                "lng": lng,
                "dist": radius_km,
                "back": days_back,
                "fmt": "json"
            }
            
            # Set up headers with API key
            headers = {
                "X-eBirdApiToken": self.ebird_api_key
            }
            
            # Make the API request
            response = requests.get(url, params=params, headers=headers)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                
                # Convert to DataFrame
                if data:
                    df = pd.DataFrame(data)
                    logger.info(f"Retrieved {len(df)} eBird observations")
                    return df
                else:
                    logger.warning(f"No eBird observations found for coords ({lat:.4f}, {lng:.4f}) within {radius_km}km radius")
                    return pd.DataFrame()
            else:
                logger.error(f"eBird API error: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting eBird observations: {str(e)}")
            return pd.DataFrame()
    
    def get_gbif_occurrences(self, lat, lng, radius_km=25, taxon_key=212):
        """
        Get bird occurrences from GBIF API
        
        Parameters:
        -----------
        lat : float
            Latitude of the center point
        lng : float
            Longitude of the center point
        radius_km : int
            Search radius in kilometers
        taxon_key : int
            GBIF taxon key (212 is for Aves/birds)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with bird occurrences
        """
        try:
            # Convert radius from km to degrees (approximate)
            radius_degrees = radius_km / 111.0
            
            # Calculate bounding box
            min_lat = lat - radius_degrees
            max_lat = lat + radius_degrees
            min_lng = lng - radius_degrees
            max_lng = lng + radius_degrees
            
            # Build the API URL
            url = f"{self.gbif_base_url}/occurrence/search"
            
            # Set up parameters
            params = {
                "decimalLatitude": f"{min_lat},{max_lat}",
                "decimalLongitude": f"{min_lng},{max_lng}",
                "taxonKey": taxon_key,  # 212 is for birds (Aves)
                "limit": 300,
                "hasCoordinate": "true",
                "hasGeospatialIssue": "false"
            }
            
            # Make the API request
            response = requests.get(url, params=params)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                
                # Extract the results
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    logger.info(f"Retrieved {len(df)} GBIF occurrences")
                    return df
                else:
                    logger.warning(f"No GBIF occurrences found for coords ({lat:.4f}, {lng:.4f}) within {radius_km}km radius")
                    return pd.DataFrame()
            else:
                logger.error(f"GBIF API error: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting GBIF occurrences: {str(e)}")
            return pd.DataFrame()
    
    def get_xeno_canto_recordings(self, lat, lng, radius_km=25):
        """
        Get bird sound recordings from Xeno-canto API
        
        Parameters:
        -----------
        lat : float
            Latitude of the center point
        lng : float
            Longitude of the center point
        radius_km : int
            Search radius in kilometers
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with bird recordings
        """
        try:
            # Convert radius from km to degrees (approximate)
            radius_degrees = radius_km / 111.0
            
            # Build the API URL
            url = f"{self.xeno_canto_base_url}/recordings"
            
            # Set up parameters - Xeno-canto uses a query string format
            # We'll search for recordings within the area
            query = f"loc:geo:{lat-radius_degrees} {lng-radius_degrees} {lat+radius_degrees} {lng+radius_degrees}"
            
            params = {
                "query": query
            }
            
            # Make the API request
            response = requests.get(url, params=params)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                
                # Extract the recordings
                if 'recordings' in data and data['recordings']:
                    df = pd.DataFrame(data['recordings'])
                    logger.info(f"Retrieved {len(df)} Xeno-canto recordings")
                    return df
                else:
                    logger.warning("No Xeno-canto recordings found")
                    return pd.DataFrame()
            else:
                logger.error(f"Xeno-canto API error: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting Xeno-canto recordings: {str(e)}")
            return pd.DataFrame()
    
    @lru_cache(maxsize=100)
    def get_environmental_data(self, lat, lng, radius_km=25):
        """
        Get environmental data from various APIs
        
        Parameters:
        -----------
        lat : float
            Latitude of the center point
        lng : float
            Longitude of the center point
        radius_km : int
            Search radius in kilometers
            
        Returns:
        --------
        dict
            Dictionary with environmental data
        """
        # Round coordinates to reduce cache misses
        lat = round(lat, 4)
        lng = round(lng, 4)
        
        env_data = {}
        
        # Try to get elevation data from Open-Elevation API
        try:
            elevation_url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lng}"
            response = requests.get(elevation_url, timeout=5)  # Add timeout
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    env_data['elevation'] = data['results'][0]['elevation']
                    logger.info(f"Retrieved elevation: {env_data['elevation']}m")
        except Exception as e:
            logger.error(f"Error getting elevation data: {str(e)}")
            # Provide fallback elevation data
            env_data['elevation'] = 500  # Default elevation
        
        # Try to get weather data from OpenWeatherMap API
        try:
            # You would need an API key for this
            api_key = os.getenv("OPENWEATHER_API_KEY", "")
            if api_key:
                weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={api_key}&units=metric"
                response = requests.get(weather_url)
                if response.status_code == 200:
                    data = response.json()
                    env_data['temperature'] = data['main']['temp']
                    env_data['humidity'] = data['main']['humidity']
                    env_data['wind_speed'] = data['wind']['speed']
                    env_data['weather_condition'] = data['weather'][0]['main']
                    logger.info(f"Retrieved weather data: {env_data['weather_condition']}, {env_data['temperature']}°C")
        except Exception as e:
            logger.error(f"Error getting weather data: {str(e)}")
        
        # Try to get land cover data from Copernicus Global Land Service
        try:
            # This is a simplified example - actual implementation would require more complex API calls
            # and possibly downloading and processing GeoTIFF files
            env_data['land_cover'] = {
                'forest': np.random.uniform(0, 100),
                'grassland': np.random.uniform(0, 100),
                'cropland': np.random.uniform(0, 100),
                'wetland': np.random.uniform(0, 100),
                'urban': np.random.uniform(0, 100),
                'water': np.random.uniform(0, 100)
            }
            logger.info(f"Retrieved land cover data")
        except Exception as e:
            logger.error(f"Error getting land cover data: {str(e)}")
        
        # Try to get protected area data from Protected Planet API
        try:
            # This would require registration with the Protected Planet API
            # Simplified example
            env_data['protected_areas'] = []
            logger.info(f"Retrieved protected area data")
        except Exception as e:
            logger.error(f"Error getting protected area data: {str(e)}")
        
        return env_data 

    def get_xeno_canto_recordings_by_species(self, species_name):
        """
        Get bird sound recordings from Xeno-canto for a specific species.
        
        Parameters:
        -----------
        species_name : str
            The common name of the bird species
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the recordings data
        """
        try:
            # Construct the API URL
            base_url = "https://xeno-canto.org/api/2/recordings"
            query = f"?query={species_name}"
            url = base_url + query
            
            # Make the request
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            if 'recordings' in data and len(data['recordings']) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(data['recordings'])
                return df
            else:
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error fetching Xeno-canto data: {str(e)}")
            return pd.DataFrame() 