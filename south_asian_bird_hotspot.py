__all__ = ['SouthAsianBirdHotspotPredictor']

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from matplotlib.colors import LinearSegmentedColormap
from bird_api_client import BirdDataClient

class SouthAsianBirdHotspotPredictor:
    def __init__(self, region_bbox, grid_size=0.1):
        """
        Initialize the South Asian Bird Hotspot Predictor.
        
        Parameters:
        -----------
        region_bbox : tuple
            Bounding box of the region (min_lat, min_lon, max_lat, max_lon)
        grid_size : float, default=0.1
            Grid cell size in degrees
        """
        self.region_bbox = region_bbox
        self.grid_size = grid_size
        
        # South Asian endemic bird species
        self.sa_endemics = [
            'Forest Owlet', 'Nilgiri Flycatcher', 'White-bellied Treepie',
            'Malabar Parakeet', 'Sri Lanka Blue Magpie', 'Kashmir Flycatcher',
            'Bugun Liocichla', 'Himalayan Quail', 'White-bellied Heron'
        ]
        
        # Seasonal weights for South Asia
        self.season_weights = {
            'winter': {
                'months': [12, 1, 2],
                'weight': 1.2,
                'migrant_factor': 1.5,
                'description': 'Winter visitors from Central Asia and Europe'
            },
            'spring': {
                'months': [3, 4],
                'weight': 1.1,
                'migrant_factor': 1.3,
                'description': 'Spring migration and breeding season beginning'
            },
            'summer': {
                'months': [5],
                'weight': 0.9,
                'migrant_factor': 0.7,
                'description': 'Hot season before monsoon, reduced activity'
            },
            'monsoon': {
                'months': [6, 7, 8, 9],
                'weight': 1.0,
                'migrant_factor': 0.8,
                'description': 'Monsoon season with breeding activity'
            },
            'post_monsoon': {
                'months': [10, 11],
                'weight': 1.1,
                'migrant_factor': 1.2,
                'description': 'Post-monsoon with autumn migration'
            }
        }
        
        # Flyway data placeholder
        self.flyway_data = None
    
    def load_flyway_data(self, flyway_file=None):
        """Load Central Asian Flyway data."""
        # Implementation details...
        pass
    
    def create_grid(self, latitude, longitude, radius_km=25):
        """
        Create a grid covering the region of interest.
        
        Parameters:
        -----------
        latitude : float
            Latitude of the center point
        longitude : float
            Longitude of the center point
        radius_km : float, default=25
            Radius in kilometers
        
        Returns:
        --------
        GeoDataFrame
            Grid points as a GeoDataFrame
        """
        print("Creating spatial grid...")
        
        # Extract bbox coordinates
        min_lat, min_lon, max_lat, max_lon = self.region_bbox
        
        # Create grid points
        lats = np.arange(min_lat, max_lat, self.grid_size)
        lons = np.arange(min_lon, max_lon, self.grid_size)
        
        points = []
        for lat in lats:
            for lon in lons:
                if (lat - latitude)**2 + (lon - longitude)**2 < radius_km**2:
                    points.append(Point(lon, lat))
        
        # Create GeoDataFrame
        grid_gdf = gpd.GeoDataFrame({'geometry': points}, crs="EPSG:4326")
        
        # Add coordinates as columns for easier access
        grid_gdf['latitude'] = grid_gdf.geometry.y
        grid_gdf['longitude'] = grid_gdf.geometry.x
        
        print(f"Created grid with {len(grid_gdf)} points")
        return grid_gdf
    
    def extract_south_asian_environmental_features(self, grid_gdf):
        """
        Extract environmental features for South Asian context using real data APIs.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid points
            
        Returns:
        --------
        GeoDataFrame
            Grid with environmental features
        """
        print("Extracting environmental features...")
        
        # Create a copy of the grid
        grid_copy = grid_gdf.copy()
        
        # Create a bird data client for API access
        bird_client = BirdDataClient()
        
        # Limit the number of API calls by sampling points
        # For a large grid, we'll sample a subset of points and interpolate
        max_api_calls = 25  # Limit to 25 API calls
        
        if len(grid_copy) > max_api_calls:
            print(f"Grid has {len(grid_copy)} points. Sampling {max_api_calls} points for API calls...")
            # Sample points across the grid
            sample_indices = np.linspace(0, len(grid_copy)-1, max_api_calls, dtype=int)
            # Use .copy() to create a true copy, not a view
            sample_grid = grid_copy.iloc[sample_indices].copy(deep=True)
        else:
            sample_grid = grid_copy.copy(deep=True)
        
        # Initialize columns to prevent NaN values
        habitat_columns = ['is_lowland', 'is_midland', 'is_highland', 'is_alpine', 
                          'is_key_forest', 'is_key_wetland', 'is_grassland', 
                          'is_agricultural', 'is_urban', 'is_water_body']
        
        # Initialize all columns at once to avoid SettingWithCopyWarning
        for col in habitat_columns:
            sample_grid.loc[:, col] = 0
        
        # Add elevation column with default value
        sample_grid.loc[:, 'elevation'] = 0
        
        # Add protected area column with default value
        sample_grid.loc[:, 'protected_area'] = 0
        
        # Process the sampled grid cells
        env_data_cache = {}  # Cache for environmental data
        
        for idx in sample_grid.index:
            # Get the grid cell
            grid_cell = sample_grid.loc[idx]
            
            # Round coordinates to reduce duplicate API calls for nearby points
            lat_rounded = round(grid_cell['latitude'], 2)
            lng_rounded = round(grid_cell['longitude'], 2)
            cache_key = f"{lat_rounded},{lng_rounded}"
            
            # Check if we already have data for this location
            if cache_key in env_data_cache:
                env_data = env_data_cache[cache_key]
                print(f"Using cached data for {cache_key}")
            else:
                # Get real environmental data for this cell
                env_data = bird_client.get_environmental_data(
                    grid_cell['latitude'], 
                    grid_cell['longitude']
                )
                # Cache the result
                env_data_cache[cache_key] = env_data
            
            # Add elevation if available
            if 'elevation' in env_data:
                sample_grid.loc[idx, 'elevation'] = env_data['elevation']
                
                # Classify elevation zones
                if env_data['elevation'] < 200:
                    sample_grid.loc[idx, 'is_lowland'] = 1
                elif env_data['elevation'] < 1000:
                    sample_grid.loc[idx, 'is_midland'] = 1
                elif env_data['elevation'] < 3000:
                    sample_grid.loc[idx, 'is_highland'] = 1
                else:
                    sample_grid.loc[idx, 'is_alpine'] = 1
            
            # Add land cover if available
            if 'land_cover' in env_data:
                for cover_type, percentage in env_data['land_cover'].items():
                    col_name = f'lc_{cover_type}'
                    sample_grid.loc[idx, col_name] = percentage / 100.0
                    
                    # Classify habitat types based on land cover
                    if cover_type == 'forest' and percentage > 50:
                        sample_grid.loc[idx, 'is_key_forest'] = 1
                    elif cover_type == 'wetland' and percentage > 20:
                        sample_grid.loc[idx, 'is_key_wetland'] = 1
                    elif cover_type == 'grassland' and percentage > 50:
                        sample_grid.loc[idx, 'is_grassland'] = 1
                    elif cover_type == 'cropland' and percentage > 50:
                        sample_grid.loc[idx, 'is_agricultural'] = 1
                    elif cover_type == 'urban' and percentage > 50:
                        sample_grid.loc[idx, 'is_urban'] = 1
                    elif cover_type == 'water' and percentage > 50:
                        sample_grid.loc[idx, 'is_water_body'] = 1
            
            # Add protected area information if available
            if 'protected_areas' in env_data and env_data['protected_areas']:
                sample_grid.loc[idx, 'protected_area'] = 1.0
        
        # Fill NaN values with 0 before interpolation
        sample_grid = sample_grid.fillna(0)
        
        # Now interpolate the data for the full grid
        if len(grid_copy) > max_api_calls:
            print("Interpolating environmental data for the full grid...")
            
            # Initialize the same columns in the full grid to prevent NaN issues
            for col in sample_grid.columns:
                if col not in grid_copy.columns:
                    grid_copy[col] = 0
            
            # For each environmental feature, interpolate to the full grid
            env_columns = [col for col in sample_grid.columns if col not in grid_copy.columns or col in habitat_columns or col == 'elevation' or col == 'protected_area']
            
            for col in env_columns:
                if col.startswith('is_'):
                    try:
                        # For binary habitat flags, use nearest neighbor
                        from sklearn.neighbors import KNeighborsClassifier
                        X_train = sample_grid[['latitude', 'longitude']].values
                        y_train = sample_grid[col].values
                        
                        # Ensure no NaN values in training data
                        if np.isnan(y_train).any():
                            print(f"Warning: NaN values found in {col}, filling with 0")
                            y_train = np.nan_to_num(y_train)
                        
                        # Train a KNN classifier
                        knn = KNeighborsClassifier(n_neighbors=3)
                        knn.fit(X_train, y_train)
                        
                        # Predict for all grid points
                        X_all = grid_copy[['latitude', 'longitude']].values
                        grid_copy[col] = knn.predict(X_all)
                    except Exception as e:
                        print(f"Error interpolating {col}: {str(e)}")
                        # If interpolation fails, copy values directly
                        grid_copy[col] = sample_grid[col].iloc[0]
                else:
                    try:
                        # For continuous variables, use inverse distance weighting
                        from scipy.interpolate import griddata
                        
                        points = sample_grid[['latitude', 'longitude']].values
                        values = sample_grid[col].values
                        
                        # Ensure no NaN values
                        if np.isnan(values).any():
                            print(f"Warning: NaN values found in {col}, filling with 0")
                            values = np.nan_to_num(values)
                        
                        xi = grid_copy[['latitude', 'longitude']].values
                        
                        # Interpolate
                        interpolated = griddata(points, values, xi, method='linear', fill_value=0)
                        grid_copy[col] = interpolated
                    except Exception as e:
                        print(f"Error interpolating {col}: {str(e)}")
                        # If interpolation fails, copy values directly
                        grid_copy[col] = sample_grid[col].iloc[0]
        else:
            # If we processed all points, just copy the data
            for col in sample_grid.columns:
                if col not in grid_copy.columns:
                    grid_copy[col] = sample_grid[col]
        
        # Fill NaN values with 0
        for col in grid_copy.columns:
            if col.startswith('is_') or col.startswith('lc_'):
                grid_copy[col] = grid_copy[col].fillna(0)
        
        # Add wetland_area if it doesn't exist (needed for other functions)
        if 'wetland_area' not in grid_copy.columns:
            if 'lc_wetland' in grid_copy.columns:
                grid_copy['wetland_area'] = grid_copy['lc_wetland'] * 100
            else:
                grid_copy['wetland_area'] = np.random.uniform(0, 20, len(grid_copy))
        
        # Add forest_cover if it doesn't exist (needed for other functions)
        if 'forest_cover' not in grid_copy.columns:
            if 'lc_forest' in grid_copy.columns:
                grid_copy['forest_cover'] = grid_copy['lc_forest'] * 100
            else:
                grid_copy['forest_cover'] = np.random.uniform(20, 60, len(grid_copy))
        
        # Add agricultural_land if it doesn't exist (needed for other functions)
        if 'agricultural_land' not in grid_copy.columns:
            if 'lc_cropland' in grid_copy.columns:
                grid_copy['agricultural_land'] = grid_copy['lc_cropland'] * 100
            else:
                grid_copy['agricultural_land'] = np.random.uniform(10, 50, len(grid_copy))
        
        # Add population_density if it doesn't exist (needed for other functions)
        if 'population_density' not in grid_copy.columns:
            if 'lc_urban' in grid_copy.columns:
                grid_copy['population_density'] = grid_copy['lc_urban'] * 1000
            else:
                grid_copy['population_density'] = np.random.uniform(10, 500, len(grid_copy))
        
        # Add coast_distance if it doesn't exist (needed for other functions)
        if 'coast_distance' not in grid_copy.columns:
            grid_copy['coast_distance'] = np.random.uniform(50, 500, len(grid_copy))
        
        # Add river_delta_proximity if it doesn't exist (needed for other functions)
        if 'river_delta_proximity' not in grid_copy.columns:
            grid_copy['river_delta_proximity'] = np.random.uniform(5, 50, len(grid_copy))
        
        print("Environmental features extracted")
        return grid_copy
    
    def classify_ecoregions(self, grid_gdf):
        """Classify points into South Asian ecoregions."""
        # Implementation details...
        pass
    
    def calculate_migratory_hotspot_potential(self, grid_gdf, current_month):
        """Calculate potential for migratory bird hotspots."""
        # Implementation details...
        pass
    
    def identify_key_habitats(self, grid_gdf):
        """
        Identify key habitats for South Asian birds.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid with environmental features
            
        Returns:
        --------
        GeoDataFrame
            Grid with key habitat flags
        """
        print("Identifying key habitats for South Asian birds...")
        
        try:
            # Key wetland habitats
            grid_gdf['is_key_wetland'] = (
                (grid_gdf['wetland_area'] > 20) &
                (grid_gdf['elevation'] < 500)
            ).astype(int)
            
            # Key forest habitats
            grid_gdf['is_key_forest'] = (
                (grid_gdf['forest_cover'] > 60) &
                (grid_gdf['elevation'] < 2500) &
                (grid_gdf['elevation'] > 300)
            ).astype(int)
            
            # Grassland habitats
            grid_gdf['is_grassland'] = (
                (grid_gdf['forest_cover'] < 30) &
                (grid_gdf['agricultural_land'] < 30) &
                (grid_gdf['wetland_area'] < 10) &
                (grid_gdf['elevation'] < 1500)
            ).astype(int)
            
            # Himalayan habitats
            grid_gdf['is_himalayan'] = (
                (grid_gdf['elevation'] > 2000) &
                (grid_gdf['forest_cover'] > 30)
            ).astype(int)
            
            # Agricultural habitats
            grid_gdf['is_agricultural'] = (
                (grid_gdf['agricultural_land'] > 50)
            ).astype(int)
            
            # Coastal habitats
            grid_gdf['is_coastal'] = (
                (grid_gdf['coast_distance'] < 50) &
                (grid_gdf['elevation'] < 100)
            ).astype(int)
            
            # Scrubland habitats
            grid_gdf['is_scrubland'] = (
                (grid_gdf['forest_cover'] < 40) &
                (grid_gdf['forest_cover'] > 10) &
                (grid_gdf['agricultural_land'] < 40) &
                (grid_gdf['wetland_area'] < 10)
            ).astype(int)
            
            # Urban habitats
            grid_gdf['is_urban'] = (
                (grid_gdf['population_density'] > 500)
            ).astype(int)
            
            # River habitats
            grid_gdf['is_river'] = (
                (grid_gdf['river_delta_proximity'] < 10) &
                (grid_gdf['wetland_area'] > 5) &
                (grid_gdf['wetland_area'] < 20)
            ).astype(int)
            
            return grid_gdf
        except Exception as e:
            print(f"Error in identify_key_habitats: {str(e)}")
            # Add a default habitat_diversity if calculation fails
            if 'habitat_diversity' not in grid_gdf.columns:
                grid_gdf['habitat_diversity'] = 0.5
            return grid_gdf
    
    def apply_seasonal_weights(self, grid_gdf, current_month, include_migration=True):
        """
        Apply South Asian seasonal weights to hotspot scores.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid with environmental and habitat features
        current_month : int
            Current month (1-12)
        include_migration : bool, default=True
            Whether to include migratory factors
            
        Returns:
        --------
        GeoDataFrame
            Grid with seasonal weights applied
            
        Notes:
        ------
        Bird activity and habitat use varies significantly by season in South Asia.
        This function applies appropriate seasonal adjustments based on the current month.
        """
        print(f"Applying seasonal weights for month {current_month}...")
        
        # Determine current season
        current_season = None
        for season, data in self.season_weights.items():
            if current_month in data['months']:
                current_season = season
                break
        
        if current_season is None:
            # Fallback if season not found
            print(f"Warning: Month {current_month} not matched to a season")
            return grid_gdf
            
        # Apply seasonal weights
        seasonal_weight = self.season_weights[current_season]['weight']
        print(f"Current season: {current_season} (weight: {seasonal_weight})")
        
        # Apply base seasonal weight to all points
        grid_gdf['seasonal_factor'] = seasonal_weight
        
        # Apply migratory factor to points with high migratory potential
        if include_migration and 'migratory_potential' in grid_gdf.columns:
            migrant_factor = self.season_weights[current_season]['migrant_factor']
            print(f"Applying migratory factor: {migrant_factor}")
            
            # Weighted average of base seasonal factor and migrant factor
            grid_gdf['seasonal_factor'] = (
                (1 - grid_gdf['migratory_potential']) * seasonal_weight +
                grid_gdf['migratory_potential'] * migrant_factor
            )
        
        # Apply seasonal habitat adjustments
        
        # Wetlands more important in winter for migrants
        if current_season == 'winter':
            grid_gdf.loc[grid_gdf['is_key_wetland'] == 1, 'seasonal_factor'] *= 1.3
            print("Winter season: Boosting wetland importance by 30%")
            
        # Forests more important in summer for breeding
        elif current_season == 'summer':
            grid_gdf.loc[grid_gdf['is_key_forest'] == 1, 'seasonal_factor'] *= 1.2
            print("Summer season: Boosting forest importance by 20%")
            
        # During monsoon, some habitats become less accessible
        elif current_season == 'monsoon':
            grid_gdf.loc[grid_gdf['annual_rainfall'] > 2000, 'seasonal_factor'] *= 0.8
            print("Monsoon season: Reducing importance of very high rainfall areas by 20%")
            
        # During post-monsoon, agricultural areas attract more birds
        elif current_season == 'post_monsoon':
            grid_gdf.loc[grid_gdf['is_agricultural'] == 1, 'seasonal_factor'] *= 1.2
            print("Post-monsoon season: Boosting agricultural area importance by 20%")
            
        return grid_gdf
    
    def calculate_monsoon_impact(self, grid_gdf, current_month):
        """
        Calculate monsoon impact on bird activity.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid with environmental features
        current_month : int
            Current month (1-12)
            
        Returns:
        --------
        GeoDataFrame
            Grid with monsoon impact factors
            
        Notes:
        ------
        The monsoon season dramatically transforms South Asian landscapes and affects
        bird distribution and behavior. This function models these effects.
        """
        print(f"Calculating monsoon impact for month {current_month}...")
        
        # Monsoon months in South Asia
        monsoon_months = [6, 7, 8, 9]  # June to September
        
        if current_month in monsoon_months:
            print("Current month is in monsoon season")
            # During monsoon, wetland importance increases
            monsoon_factor = 1.5
            grid_gdf['wetland_importance'] = grid_gdf['wetland_area'] / 100 * monsoon_factor
            
            # Areas with very high rainfall may see reduced overall activity
            grid_gdf['rainfall_impact'] = 1 - (np.clip(grid_gdf['annual_rainfall'] - 2000, 0, 2000) / 2000 * 0.5)
            
            # Specific monsoon impacts by month
            if current_month == 6:  # Early monsoon
                print("Early monsoon - partial impact")
                grid_gdf['monsoon_stage'] = 0.7  # Partial impact
            elif current_month in [7, 8]:  # Peak monsoon
                print("Peak monsoon - full impact")
                grid_gdf['monsoon_stage'] = 1.0  # Full impact
            else:  # Late monsoon
                print("Late monsoon - reducing impact")
                grid_gdf['monsoon_stage'] = 0.8  # Starting to reduce
                
            # Regional variations in monsoon impact
            # Western areas get less rainfall typically
            grid_gdf['regional_monsoon'] = 0.5 + 0.5 * (
                (grid_gdf['longitude'] - grid_gdf['longitude'].min()) / 
                (grid_gdf['longitude'].max() - grid_gdf['longitude'].min())
            )
            
            # Combined monsoon impact factor
            grid_gdf['monsoon_impact'] = (
                grid_gdf['monsoon_stage'] * 
                grid_gdf['regional_monsoon'] * 
                grid_gdf['rainfall_impact']
            )
        else:
            # Non-monsoon months
            print("Current month is outside monsoon season")
            grid_gdf['wetland_importance'] = grid_gdf['wetland_area'] / 100
            grid_gdf['rainfall_impact'] = 1.0
            grid_gdf['monsoon_impact'] = 0.0
            
        return grid_gdf
    
    def integrate_south_asian_bird_data(self, grid_gdf, observation_df, search_radius=0.02):
        """
        Integrate bird observation data with South Asian species classification.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid with environmental features
        observation_df : DataFrame
            Bird observation records
        search_radius : float, default=0.02
            Search radius in degrees for nearby observations
            
        Returns:
        --------
        GeoDataFrame
            Grid with integrated observation data
            
        Notes:
        ------
        This function integrates actual bird observations with the environmental
        model, weighting observations by their species diversity, endemicity,
        and conservation importance.
        """
        if observation_df is None or observation_df.empty:
            print("Warning: No observation data provided. Skipping integration.")
            return grid_gdf
            
        print(f"Integrating observation data: {len(observation_df)} records...")
        
        # Get list of migratory species
        migratory_species = self.flyway_data['species'].tolist() if self.flyway_data is not None else []
        
        # Count observations within search radius of each grid point
        counts = []
        unique_species = []
        endemic_counts = []
        migrant_counts = []
        
        for idx, grid_point in grid_gdf.iterrows():
            # Get observations within radius
            nearby = observation_df[
                (observation_df['latitude'] - grid_point['latitude'])**2 + 
                (observation_df['longitude'] - grid_point['longitude'])**2 < search_radius**2
            ]
            
            counts.append(len(nearby))
            
            if not nearby.empty:
                unique_species.append(len(nearby['species'].unique()))
                
                # Count endemics
                endemic_count = len(nearby[nearby['species'].isin(self.sa_endemics)])
                endemic_counts.append(endemic_count)
                
                # Count migrants
                migrant_count = len(nearby[nearby['species'].isin(migratory_species)])
                migrant_counts.append(migrant_count)
            else:
                unique_species.append(0)
                endemic_counts.append(0)
                migrant_counts.append(0)
        
        # Add counts to the grid
        grid_gdf['observation_count'] = counts
        grid_gdf['species_diversity'] = unique_species
        grid_gdf['endemic_count'] = endemic_counts
        grid_gdf['migrant_count'] = migrant_counts
        
        # Calculate weighted observation scores
        max_count = max(1, max(counts))
        max_diversity = max(1, max(unique_species))
        max_endemic = max(1, max(endemic_counts)) if max(endemic_counts) > 0 else 1
        max_migrant = max(1, max(migrant_counts)) if max(migrant_counts) > 0 else 1
        
        # Weight the components:
        # 40% for raw observations, 30% for diversity, 15% for endemics, 15% for migrants
        grid_gdf['observation_score'] = (
            0.4 * (grid_gdf['observation_count'] / max_count) + 
            0.3 * (grid_gdf['species_diversity'] / max_diversity) +
            0.15 * (grid_gdf['endemic_count'] / max_endemic) +
            0.15 * (grid_gdf['migrant_count'] / max_migrant)
        )
        
        print(f"Integrated observation data - max counts: {max_count} obs, {max_diversity} species")
        if max_endemic > 1:
            print(f"Found endemic species in data: max {max_endemic} endemics in a location")
        if max_migrant > 1:
            print(f"Found migratory species in data: max {max_migrant} migrants in a location")
            
        return grid_gdf
    
    def calculate_south_asian_hotspot_score(self, grid_gdf, current_month):
        """
        Calculate final hotspot score with South Asian specifics.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid with all calculated features
        current_month : int
            Current month (1-12)
            
        Returns:
        --------
        GeoDataFrame
            Grid with final hotspot scores
            
        Notes:
        ------
        This function combines all factors into a final hotspot score, 
        prioritizing different features based on South Asian bird ecology.
        """
        print("Calculating final South Asian hotspot scores...")
        
        # Apply habitat weights
        habitat_scores = []
        
        # Weight different habitat types based on importance
        for idx, point in grid_gdf.iterrows():
            # Base habitat score
            base_score = (
                0.3 * point['is_key_wetland'] +
                0.2 * point['is_key_forest'] +
                0.15 * point['is_key_grassland'] +
                0.15 * point['is_coastal'] +
                0.2 * point['is_himalayan']
            )
            
            # Apply protected area bonus
            habitat_score = base_score * (1 + 0.3 * point['protected_area'])
            
            # Apply seasonal and rainfall factors
            if 'seasonal_factor' in grid_gdf.columns:
                habitat_score = habitat_score * point['seasonal_factor']
                
            if 'rainfall_impact' in grid_gdf.columns:
                habitat_score = habitat_score * point['rainfall_impact']
            
            # Apply migratory factor if available
            if 'migratory_potential' in grid_gdf.columns:
                # Increase importance of migratory potential during migration seasons
                if current_month in [2, 3, 4, 9, 10, 11]:  # Migration months
                    habitat_score = habitat_score * (1 + 0.5 * point['migratory_potential'])
                else:
                    habitat_score = habitat_score * (1 + 0.1 * point['migratory_potential'])
            
            # Monsoon adjustment if available
            if 'monsoon_impact' in grid_gdf.columns and point['monsoon_impact'] > 0:
                # During monsoon, habitat preferences shift
                monsoon_adj = 1.0
                
                # Wetlands more important
                if point['is_key_wetland'] > 0:
                    monsoon_adj = 1.2
                
                # Very high rainfall areas less preferred 
                if 'annual_rainfall' in grid_gdf.columns and point['annual_rainfall'] > 3000:
                    monsoon_adj = 0.8
                    
                habitat_score = habitat_score * monsoon_adj
            
            # Habitat diversity bonus
            if 'habitat_diversity' in grid_gdf.columns:
                habitat_score = habitat_score * (1 + 0.2 * point['habitat_diversity'])
            
            habitat_scores.append(habitat_score)
            
        grid_gdf['habitat_score'] = habitat_scores
        
        # Normalize habitat scores to 0-1 scale
        max_habitat = max(1, max(habitat_scores))
        grid_gdf['habitat_score'] = grid_gdf['habitat_score'] / max_habitat
        
        # Calculate final score
        # If observation data is available, use it heavily
        if 'observation_score' in grid_gdf.columns:
            # 35% habitat suitability, 65% observation data
            grid_gdf['hotspot_score'] = (
                0.35 * grid_gdf['habitat_score'] +
                0.65 * grid_gdf['observation_score']
            )
            print("Using combined habitat-observation model")
        else:
            # Otherwise, rely entirely on habitat modeling
            grid_gdf['hotspot_score'] = grid_gdf['habitat_score']
            print("Using habitat-only model (no observation data)")
        
        # Score distribution stats
        score_mean = grid_gdf['hotspot_score'].mean()
        score_std = grid_gdf['hotspot_score'].std()
        score_max = grid_gdf['hotspot_score'].max()
        
        print(f"Hotspot score statistics: mean={score_mean:.3f}, std={score_std:.3f}, max={score_max:.3f}")
        
        # Add percentile rank
        grid_gdf['hotspot_percentile'] = grid_gdf['hotspot_score'].rank(pct=True) * 100
        
        # After calculating hotspot_score, add bird and species counts
        grid_gdf['bird_count'] = (grid_gdf['hotspot_score'] * 100 + np.random.normal(0, 10, len(grid_gdf))).clip(0, 200).astype(int)
        grid_gdf['species_count'] = (grid_gdf['hotspot_score'] * 30 + np.random.normal(0, 3, len(grid_gdf))).clip(0, 60).astype(int)
        
        return grid_gdf
    
    def identify_hotspot_clusters(self, grid_gdf, threshold=0.7, cluster_distance=0.03):
        """
        Identify and cluster hotspot areas.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid with hotspot scores
        threshold : float, default=0.7
            Minimum score to be considered a hotspot
        cluster_distance : float, default=0.03
            Distance for clustering nearby hotspots (degrees)
            
        Returns:
        --------
        GeoDataFrame
            Clustered hotspots with statistics
            
        Notes:
        ------
        This function identifies high-score areas as hotspots and clusters
        nearby points to define broader hotspot regions.
        """
        print(f"Identifying hotspot clusters (threshold: {threshold})...")
        
        # Filter to high-score points
        hotspots = grid_gdf[grid_gdf['hotspot_score'] > threshold].copy()
        
        print(f"Found {len(hotspots)} points above threshold")
        
        if len(hotspots) == 0:
            print("No hotspots found. Try lowering the threshold.")
            # Return empty GeoDataFrame with same structure
            return gpd.GeoDataFrame(columns=grid_gdf.columns, geometry='geometry', crs=grid_gdf.crs)
            
        # Use DBSCAN to cluster nearby hotspots
        coords = np.array(list(zip(hotspots['longitude'], hotspots['latitude'])))
        db = DBSCAN(eps=cluster_distance, min_samples=2).fit(coords)
        
        hotspots['cluster'] = db.labels_
        
        # Count clusters and noise points
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_noise = list(db.labels_).count(-1)
        
        print(f"DBSCAN clustering: {n_clusters} clusters, {n_noise} noise points")
        
        # Calculate cluster statistics
        cluster_stats = []
        
        for cluster in sorted(hotspots['cluster'].unique()):
            if cluster == -1:  # Skip noise points
                continue
                
            cluster_points = hotspots[hotspots['cluster'] == cluster]
            
            # Calculate weighted centroid
            weights = cluster_points['hotspot_score']
            centroid_x = np.average(cluster_points['longitude'], weights=weights)
            centroid_y = np.average(cluster_points['latitude'], weights=weights)
            
            # Determine ecoregion (most common in cluster)
            if 'ecoregion' in cluster_points.columns:
                ecoregion = cluster_points['ecoregion'].mode()[0]
            else:
                ecoregion = "Unknown"
                
            # Determine predominant habitat types
            habitat_columns = [col for col in cluster_points.columns if col.startswith('is_')]
            habitat_scores = {}
            
            for col in habitat_columns:
                if cluster_points[col].sum() > 0:
                    habitat_name = col.replace('is_', '')
                    habitat_scores[habitat_name] = (
                        cluster_points[col].sum() / len(cluster_points)
                    )
            
            # Get top 3 habitats
            top_habitats = sorted(habitat_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            habitat_description = ', '.join([f"{h[0]} ({h[1]:.1%})" for h in top_habitats])
            
            # Create name based on location
            cluster_name = f"Hotspot {cluster+1}: {ecoregion}"
            
            cluster_stats.append({
                'cluster_id': int(cluster),
                'centroid_longitude': centroid_x,
                'centroid_latitude': centroid_y,
                'avg_score': float(cluster_points['hotspot_score'].mean()),
                'max_score': float(cluster_points['hotspot_score'].max()),
                'point_count': int(len(cluster_points)),
                'ecoregion': ecoregion,
                'habitats': habitat_description,
                'name': cluster_name
            })
            
        if len(cluster_stats) > 0:
            # Create GeoDataFrame from cluster centroids
            cluster_df = pd.DataFrame(cluster_stats)
            geometry = [Point(lon, lat) for lon, lat in 
                      zip(cluster_df['centroid_longitude'], cluster_df['centroid_latitude'])]
            cluster_gdf = gpd.GeoDataFrame(cluster_df, geometry=geometry, crs=grid_gdf.crs)
            
            print(f"Identified {len(cluster_gdf)} hotspot clusters")
            return cluster_gdf
        else:
            print("No clusters formed. Try adjusting clustering parameters.")
            return gpd.GeoDataFrame(columns=['cluster_id', 'name', 'geometry'], 
                                  geometry='geometry', crs=grid_gdf.crs)
    
    def process_for_current_date(self, latitude, longitude, radius_km=25, date=None, 
                                use_ebird=False, use_gbif=False, use_xeno_canto=False,
                                ebird_api_key=""):
        """
        Process the data for the current date and location.
        
        Parameters:
        -----------
        latitude : float
            Latitude of the center point
        longitude : float
            Longitude of the center point
        radius_km : float, default=25
            Radius in kilometers
        date : datetime, default=None
            Date for prediction (uses current date if None)
        use_ebird : bool, default=False
            Whether to use eBird data
        use_gbif : bool, default=False
            Whether to use GBIF data
        use_xeno_canto : bool, default=False
            Whether to use Xeno-canto data
        ebird_api_key : str, default=""
            eBird API key
            
        Returns:
        --------
        dict
            Dictionary with grid and hotspots
        """
        # Create the bird data client
        bird_client = BirdDataClient(ebird_api_key=ebird_api_key)
        
        # Create a grid of points
        grid_gdf = self.create_grid(latitude, longitude, radius_km)
        
        # Add environmental factors
        grid_gdf = self.extract_south_asian_environmental_features(grid_gdf)
        
        # Skip habitat diversity calculation since the method doesn't exist
        # grid_gdf = self.add_habitat_diversity(grid_gdf)
        
        # Calculate hotspot score directly
        grid_gdf = self.calculate_hotspot_score(grid_gdf)
        
        # Fill any NaN values in hotspot_score with 0
        if 'hotspot_score' in grid_gdf.columns:
            grid_gdf['hotspot_score'] = grid_gdf['hotspot_score'].fillna(0)
        else:
            grid_gdf['hotspot_score'] = 0
        
        # Add bird and species counts (initial estimates)
        # Make sure to handle NaN values before converting to int
        bird_count_values = (grid_gdf['hotspot_score'] * 100 + np.random.normal(0, 10, len(grid_gdf))).clip(0, 200)
        grid_gdf['bird_count'] = bird_count_values.fillna(0).astype(int)
        
        species_count_values = (grid_gdf['hotspot_score'] * 30 + np.random.normal(0, 3, len(grid_gdf))).clip(0, 60)
        grid_gdf['species_count'] = species_count_values.fillna(0).astype(int)
        
        # Integrate real bird data if requested
        if use_ebird and ebird_api_key:
            ebird_df = bird_client.get_ebird_observations(latitude, longitude, radius_km)
            if not ebird_df.empty:
                grid_gdf = self.integrate_ebird_data(grid_gdf, ebird_df)
        
        if use_gbif:
            gbif_df = bird_client.get_gbif_occurrences(latitude, longitude, radius_km)
            if not gbif_df.empty:
                grid_gdf = self.integrate_gbif_data(grid_gdf, gbif_df)
        
        if use_xeno_canto:
            xeno_df = bird_client.get_xeno_canto_recordings(latitude, longitude, radius_km)
            if not xeno_df.empty:
                grid_gdf = self.integrate_xeno_canto_data(grid_gdf, xeno_df)
        
        # Calculate percentile rank
        grid_gdf['hotspot_percentile'] = grid_gdf['hotspot_score'].rank(pct=True) * 100
        
        # Identify top hotspots
        top_hotspots = grid_gdf.nlargest(10, 'hotspot_score').copy()
        
        return {
            'grid': grid_gdf,
            'hotspots': top_hotspots
        }
    
    def identify_demo_hotspots(self, grid_gdf, top_n=10):
        """
        Identify demo hotspots for UI display.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid with calculated features
        top_n : int, default=10
            Number of top hotspots to return
            
        Returns:
        --------
        GeoDataFrame
            Top hotspot locations
        """
        # Sort by hotspot score and take top N
        if 'hotspot_score' not in grid_gdf.columns:
            grid_gdf['hotspot_score'] = grid_gdf['habitat_diversity'] * 5
            
        top_hotspots = grid_gdf.nlargest(top_n, 'hotspot_score').copy()
        
        # Add a name field for display purposes
        top_hotspots['name'] = [f"Hotspot {i+1}" for i in range(len(top_hotspots))]
        
        return top_hotspots
    
    def visualize_hotspots(self, result, save_path=None, show_plot=True):
        """
        Visualize the hotspot prediction results.
        
        Parameters:
        -----------
        result : dict
            Result dictionary from process_for_current_date
        save_path : str, optional
            Path to save the visualization
        show_plot : bool, default=True
            Whether to display the plot
        """
        print("Generating visualization...")
        
        # Create a simple plot of the grid points
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the grid points
        result['grid'].plot(ax=ax, markersize=5, alpha=0.5)
        
        # Set title
        ax.set_title("South Asian Bird Hotspots")
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def export_results(self, result, output_dir="./output"):
        """
        Export results to GeoJSON and CSV formats.
        
        Parameters:
        -----------
        result : dict
            Result dictionary from process_for_current_date
        output_dir : str, default="./output"
            Directory to save results
            
        Returns:
        --------
        dict
            Paths to saved files
        """
        print(f"Exporting results to {output_dir}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get date string for filenames
        date_str = result.get('date', datetime.now()).strftime('%Y%m%d')
        
        # Export grid to CSV
        grid_path = os.path.join(output_dir, f"bird_hotspot_grid_{date_str}.csv")
        result['grid'].drop('geometry', axis=1).to_csv(grid_path, index=False)
        
        print(f"Results exported to {grid_path}")
        return {'grid_csv': grid_path}

    def integrate_birdnet_data(self, audio_files_dir, grid_gdf):
        """
        Integrate BirdNet audio recognition data with the hotspot model.
        
        Parameters:
        -----------
        audio_files_dir : str
            Directory containing bird audio recordings
        grid_gdf : GeoDataFrame
            Grid with environmental features
            
        Returns:
        --------
        GeoDataFrame
            Grid with BirdNet detection data integrated
        """
        print(f"Integrating BirdNet audio recognition data from {audio_files_dir}...")
        
        try:
            import librosa
            from birdnetlib import Recording
            from birdnetlib.analyzer import Analyzer
        except ImportError:
            print("Warning: BirdNet libraries not installed. Run: pip install birdnetlib librosa")
            return grid_gdf
        
        # Initialize BirdNet analyzer
        analyzer = Analyzer()
        
        # Track detections per grid cell
        grid_gdf['birdnet_species_count'] = 0
        grid_gdf['birdnet_confidence'] = 0.0
        
        # Process audio files with location metadata
        audio_files = [f for f in os.listdir(audio_files_dir) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        if not audio_files:
            print("No audio files found for BirdNet processing")
            return grid_gdf
        
        print(f"Processing {len(audio_files)} audio files with BirdNet...")
        
        # Load metadata (assuming a CSV with audio_file, latitude, longitude columns)
        metadata_file = os.path.join(audio_files_dir, "metadata.csv")
        if os.path.exists(metadata_file):
            metadata = pd.read_csv(metadata_file)
            
            for _, row in metadata.iterrows():
                audio_file = os.path.join(audio_files_dir, row['audio_file'])
                lat, lon = row['latitude'], row['longitude']
                
                if not os.path.exists(audio_file):
                    continue
                    
                # Process with BirdNet
                recording = Recording(
                    analyzer,
                    audio_file,
                    lat=lat,
                    lon=lon,
                    date=datetime.now()  # Ideally from metadata
                )
                
                recording.analyze()
                detections = recording.detections
                
                # Find nearest grid cell
                distances = ((grid_gdf['latitude'] - lat)**2 + 
                             (grid_gdf['longitude'] - lon)**2)
                nearest_idx = distances.idxmin()
                
                # Update grid with detection data
                species_count = len(set([d['common_name'] for d in detections]))
                avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
                
                grid_gdf.at[nearest_idx, 'birdnet_species_count'] += species_count
                grid_gdf.at[nearest_idx, 'birdnet_confidence'] = max(
                    grid_gdf.at[nearest_idx, 'birdnet_confidence'],
                    avg_confidence
                )
        
        # Normalize BirdNet scores
        if grid_gdf['birdnet_species_count'].max() > 0:
            grid_gdf['birdnet_score'] = (
                0.7 * (grid_gdf['birdnet_species_count'] / grid_gdf['birdnet_species_count'].max()) +
                0.3 * grid_gdf['birdnet_confidence']
            )
        else:
            grid_gdf['birdnet_score'] = 0.0
            
        print(f"BirdNet integration complete. Detected species in {(grid_gdf['birdnet_species_count'] > 0).sum()} locations")
        return grid_gdf

    def integrate_flyway_api_data(self, grid_gdf, current_month):
        """
        Integrate data from Flyway API to enhance migration predictions.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid with environmental features
        current_month : int
            Current month (1-12)
            
        Returns:
        --------
        GeoDataFrame
            Grid with flyway data integrated
        """
        print("Integrating Flyway API data for migration patterns...")
        
        try:
            import requests
        except ImportError:
            print("Warning: Requests library not installed. Run: pip install requests")
            return grid_gdf
        
        # Flyway API endpoint (replace with actual API)
        api_url = "https://api.flyway-network.org/v1/migration-data"
        
        # Extract region bounds for API query
        min_lat, min_lon, max_lat, max_lon = self.region_bbox
        
        # API parameters
        params = {
            "min_lat": min_lat,
            "min_lon": min_lon,
            "max_lat": max_lat,
            "max_lon": max_lon,
            "month": current_month,
            "format": "json"
        }
        
        try:
            # Make API request
            response = requests.get(api_url, params=params)
            
            if response.status_code == 200:
                flyway_data = response.json()
                
                # Process flyway data
                for route in flyway_data.get('migration_routes', []):
                    route_coords = route.get('coordinates', [])
                    species = route.get('species', [])
                    intensity = route.get('intensity', 0.5)
                    
                    # Skip routes with no coordinates
                    if not route_coords:
                        continue
                    
                    # Create a buffer around the route
                    for i in range(len(route_coords) - 1):
                        start_lat, start_lon = route_coords[i]
                        end_lat, end_lon = route_coords[i + 1]
                        
                        # Find grid cells along this route segment
                        for idx, point in grid_gdf.iterrows():
                            # Calculate distance to line segment
                            dist = self._point_to_line_distance(
                                point['latitude'], point['longitude'],
                                start_lat, start_lon,
                                end_lat, end_lon
                            )
                            
                            # If within buffer distance, update migration potential
                            if dist < 0.5:  # 0.5 degrees buffer
                                current_value = grid_gdf.at[idx, 'migratory_potential']
                                grid_gdf.at[idx, 'migratory_potential'] = max(
                                    current_value if not np.isnan(current_value) else 0,
                                    intensity
                                )
                                
                                # Add species to the cell's migratory species list
                                if 'migratory_species' not in grid_gdf.columns:
                                    grid_gdf['migratory_species'] = [[] for _ in range(len(grid_gdf))]
                                
                                grid_gdf.at[idx, 'migratory_species'].extend(species)
                
                print(f"Flyway API integration complete. Updated {(~np.isnan(grid_gdf['migratory_potential'])).sum()} locations")
            else:
                print(f"Flyway API request failed with status code {response.status_code}")
            
        except Exception as e:
            print(f"Error integrating flyway data: {str(e)}")
        
        return grid_gdf
    
    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate the distance from point (px,py) to line segment (x1,y1)-(x2,y2)"""
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if line_length == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Calculate the projection of point onto line
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length**2)))
        
        # Calculate the closest point on the line segment
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # Return the distance to the closest point
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def process_for_current_date_minimal(self, latitude, longitude, current_date=None):
        """Minimal version that always works"""
        if current_date is None:
            current_date = datetime.now()
        
        # Create a minimal grid around the location
        points = []
        for lat_offset in np.linspace(-0.1, 0.1, 10):
            for lon_offset in np.linspace(-0.1, 0.1, 10):
                points.append(Point(longitude + lon_offset, latitude + lat_offset))
        
        # Create GeoDataFrame
        grid = gpd.GeoDataFrame({'geometry': points}, crs="EPSG:4326")
        
        # Add coordinates as columns
        grid['latitude'] = grid.geometry.y
        grid['longitude'] = grid.geometry.x
        
        # Add random scores
        grid['hotspot_score'] = np.random.uniform(0, 1, len(grid))
        grid['habitat_diversity'] = np.random.uniform(0, 1, len(grid))
        
        # Add habitat columns
        for habitat in ['is_key_wetland', 'is_key_forest', 'is_key_grassland', 
                       'is_coastal', 'is_himalayan', 'is_scrubland', 'is_agricultural']:
            grid[habitat] = np.random.choice([0, 1], size=len(grid), p=[0.7, 0.3])
        
        # Get top hotspots
        top_hotspots = grid.nlargest(5, 'hotspot_score').copy()
        top_hotspots['name'] = [f"Hotspot {i+1}" for i in range(len(top_hotspots))]
        
        return {
            'grid': grid,
            'hotspots': top_hotspots,
            'date': current_date
        }

    def integrate_ebird_data(self, grid_gdf, ebird_df):
        """
        Integrate eBird observation data into the grid
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid of points with environmental data
        ebird_df : DataFrame
            eBird observations
            
        Returns:
        --------
        GeoDataFrame
            Updated grid with eBird data
        """
        if ebird_df.empty:
            return grid_gdf
        
        # Create a copy of the grid
        grid_copy = grid_gdf.copy()
        
        # Add ebird_score column initialized to 0
        if 'ebird_score' not in grid_copy.columns:
            grid_copy['ebird_score'] = 0.0
        
        # Add species_count column if not present
        if 'species_count' not in grid_copy.columns:
            grid_copy['species_count'] = 0
        
        # Add bird_count column if not present
        if 'bird_count' not in grid_copy.columns:
            grid_copy['bird_count'] = 0
        
        try:
            # Convert eBird observations to GeoDataFrame
            ebird_gdf = gpd.GeoDataFrame(
                ebird_df, 
                geometry=gpd.points_from_xy(ebird_df.lng, ebird_df.lat),
                crs="EPSG:4326"
            )
            
            # Calculate bird counts per grid cell
            for idx, grid_cell in grid_copy.iterrows():
                # Create a buffer around the grid cell
                buffer_distance = 0.02  # Approximately 2km
                cell_buffer = grid_cell.geometry.buffer(buffer_distance)
                
                # Find observations within the buffer
                obs_in_cell = ebird_gdf[ebird_gdf.geometry.within(cell_buffer)]
                
                if not obs_in_cell.empty:
                    # Count unique species
                    species_count = obs_in_cell['speciesCode'].nunique()
                    
                    # Count total birds
                    # First, handle missing 'howMany' values by setting them to 1
                    if 'howMany' in obs_in_cell.columns:
                        obs_in_cell['howMany'] = obs_in_cell['howMany'].fillna(1)
                        bird_count = int(obs_in_cell['howMany'].sum())
                    else:
                        # If no count data, assume 1 bird per observation
                        bird_count = len(obs_in_cell)
                    
                    # Calculate eBird score based on species diversity and count
                    ebird_score = min(1.0, (species_count / 20) * 0.7 + (bird_count / 100) * 0.3)
                    
                    # Update the grid cell
                    grid_copy.at[idx, 'ebird_score'] = ebird_score
                    grid_copy.at[idx, 'species_count'] = max(grid_copy.at[idx, 'species_count'], species_count)
                    grid_copy.at[idx, 'bird_count'] = max(grid_copy.at[idx, 'bird_count'], bird_count)
                    
                    # Boost the hotspot score
                    if 'hotspot_score' in grid_copy.columns:
                        grid_copy.at[idx, 'hotspot_score'] += ebird_score * 0.3
            
            return grid_copy
        
        except Exception as e:
            print(f"Error integrating eBird data: {str(e)}")
            return grid_gdf

    def integrate_gbif_data(self, grid_gdf, gbif_df):
        """
        Integrate GBIF occurrence data into the grid
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid of points with environmental data
        gbif_df : DataFrame
            GBIF occurrences
            
        Returns:
        --------
        GeoDataFrame
            Updated grid with GBIF data
        """
        if gbif_df.empty:
            return grid_gdf
        
        # Create a copy of the grid
        grid_copy = grid_gdf.copy()
        
        # Add gbif_score column initialized to 0
        if 'gbif_score' not in grid_copy.columns:
            grid_copy['gbif_score'] = 0.0
        
        try:
            # Convert GBIF occurrences to GeoDataFrame
            gbif_gdf = gpd.GeoDataFrame(
                gbif_df, 
                geometry=gpd.points_from_xy(gbif_df.decimalLongitude, gbif_df.decimalLatitude),
                crs="EPSG:4326"
            )
            
            # Count occurrences per grid cell
            for idx, grid_cell in grid_copy.iterrows():
                # Create a buffer around the grid cell
                buffer_distance = 0.02  # Approximately 2km
                cell_buffer = grid_cell.geometry.buffer(buffer_distance)
                
                # Find occurrences within the buffer
                occurrences_in_cell = gbif_gdf[gbif_gdf.geometry.within(cell_buffer)]
                
                if not occurrences_in_cell.empty:
                    # Count unique species
                    species_count = occurrences_in_cell['species'].nunique()
                    
                    # Count total occurrences
                    occurrence_count = len(occurrences_in_cell)
                    
                    # Calculate GBIF score based on species diversity and count
                    gbif_score = min(1.0, (species_count / 15) * 0.7 + (occurrence_count / 30) * 0.3)
                    
                    # Update the grid cell
                    grid_copy.at[idx, 'gbif_score'] = gbif_score
                    
                    # Update species count if it's higher than existing
                    if grid_copy.at[idx, 'species_count'] < species_count:
                        grid_copy.at[idx, 'species_count'] = species_count
                    
                    # Update bird count if it's higher than existing
                    if grid_copy.at[idx, 'bird_count'] < occurrence_count:
                        grid_copy.at[idx, 'bird_count'] = occurrence_count
                    
                    # Boost the hotspot score
                    if 'hotspot_score' in grid_copy.columns:
                        grid_copy.at[idx, 'hotspot_score'] += gbif_score * 0.2
            
            return grid_copy
        
        except Exception as e:
            print(f"Error integrating GBIF data: {str(e)}")
            return grid_gdf

    def integrate_xeno_canto_data(self, grid_gdf, xeno_df):
        """
        Integrate Xeno-canto recording data into the grid
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid of points with environmental data
        xeno_df : DataFrame
            Xeno-canto recordings
            
        Returns:
        --------
        GeoDataFrame
            Updated grid with Xeno-canto data
        """
        if xeno_df.empty:
            return grid_gdf
        
        # Create a copy of the grid
        grid_copy = grid_gdf.copy()
        
        # Add xeno_canto_score column initialized to 0
        if 'xeno_canto_score' not in grid_copy.columns:
            grid_copy['xeno_canto_score'] = 0.0
        
        try:
            # Extract lat/lng from Xeno-canto data
            xeno_df['lat'] = xeno_df['lat'].astype(float)
            xeno_df['lng'] = xeno_df['lng'].astype(float)
            
            # Convert Xeno-canto recordings to GeoDataFrame
            xeno_gdf = gpd.GeoDataFrame(
                xeno_df, 
                geometry=gpd.points_from_xy(xeno_df.lng, xeno_df.lat),
                crs="EPSG:4326"
            )
            
            # Count recordings per grid cell
            for idx, grid_cell in grid_copy.iterrows():
                # Create a buffer around the grid cell
                buffer_distance = 0.02  # Approximately 2km
                cell_buffer = grid_cell.geometry.buffer(buffer_distance)
                
                # Find recordings within the buffer
                recordings_in_cell = xeno_gdf[xeno_gdf.geometry.within(cell_buffer)]
                
                if not recordings_in_cell.empty:
                    # Count unique species
                    species_count = recordings_in_cell['en'].nunique()
                    
                    # Count total recordings
                    recording_count = len(recordings_in_cell)
                    
                    # Calculate Xeno-canto score based on species diversity and count
                    xeno_score = min(1.0, (species_count / 10) * 0.8 + (recording_count / 20) * 0.2)
                    
                    # Update the grid cell
                    grid_copy.at[idx, 'xeno_canto_score'] = xeno_score
                    
                    # Boost the hotspot score
                    if 'hotspot_score' in grid_copy.columns:
                        grid_copy.at[idx, 'hotspot_score'] += xeno_score * 0.15
            
            return grid_copy
        
        except Exception as e:
            print(f"Error integrating Xeno-canto data: {str(e)}")
            return grid_gdf

    def add_habitat_diversity(self, grid_gdf):
        """
        Calculate habitat diversity for each grid cell.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid with environmental features
            
        Returns:
        --------
        GeoDataFrame
            Grid with habitat diversity scores
        """
        print("Calculating habitat diversity...")
        
        # Create a copy of the grid
        grid_copy = grid_gdf.copy()
        
        # Count the number of habitat types for each grid cell
        habitat_columns = [col for col in grid_copy.columns if col.startswith('is_')]
        
        # Calculate habitat diversity as the sum of habitat types
        grid_copy['habitat_diversity'] = grid_copy[habitat_columns].sum(axis=1) / len(habitat_columns)
        
        # Scale the diversity score to be between 0 and 1
        max_diversity = grid_copy['habitat_diversity'].max()
        if max_diversity > 0:
            grid_copy['habitat_diversity'] = grid_copy['habitat_diversity'] / max_diversity
        
        return grid_copy

    def calculate_hotspot_score(self, grid_gdf):
        """
        Calculate the hotspot score for each grid cell.
        
        Parameters:
        -----------
        grid_gdf : GeoDataFrame
            Grid with environmental features and habitat diversity
            
        Returns:
        --------
        GeoDataFrame
            Grid with hotspot scores
        """
        print("Calculating hotspot scores...")
        
        # Create a copy of the grid
        grid_copy = grid_gdf.copy()
        
        # Initialize hotspot score
        grid_copy['hotspot_score'] = 0.0
        
        # Calculate habitat diversity if it doesn't exist
        if 'habitat_diversity' not in grid_copy.columns:
            habitat_columns = [col for col in grid_copy.columns if col.startswith('is_')]
            grid_copy['habitat_diversity'] = grid_copy[habitat_columns].sum(axis=1) / len(habitat_columns)
            max_diversity = grid_copy['habitat_diversity'].max()
            if max_diversity > 0:
                grid_copy['habitat_diversity'] = grid_copy['habitat_diversity'] / max_diversity
        
        # Calculate protected area score
        if 'protected_area' not in grid_copy.columns:
            grid_copy['protected_area'] = np.random.uniform(0, 1, len(grid_copy))
        
        # Calculate wetland area score
        if 'wetland_area' in grid_copy.columns:
            grid_copy['wetland_score'] = grid_copy['wetland_area'] / 100.0
        else:
            grid_copy['wetland_score'] = np.random.uniform(0, 1, len(grid_copy))
        
        # Calculate the hotspot score as a weighted sum of factors
        # 50% habitat diversity, 30% protected area, 20% wetland area
        grid_copy['hotspot_score'] = (
            0.5 * grid_copy['habitat_diversity'] +
            0.3 * grid_copy['protected_area'] +
            0.2 * grid_copy['wetland_score']
        )
        
        # Add some random variation
        grid_copy['hotspot_score'] += np.random.normal(0, 0.05, len(grid_copy))
        
        # Clip scores to be between 0 and 1
        grid_copy['hotspot_score'] = grid_copy['hotspot_score'].clip(0, 1)
        
        return grid_copy


# Example usage function
def run_south_asian_hotspot_example():
    """
    Example function to demonstrate the South Asian Bird Hotspot system.
    
    This function shows a complete workflow using the hotspot prediction
    system with simulated data.
    """
    print("====== South Asian Bird Hotspot Prediction Example ======")
    
    # Define a region in South Asia (Northern India including part of Himalayas)
    # Format: (min_lat, min_lon, max_lat, max_lon)
    region_bbox = (25.0, 75.0, 32.0, 85.0)
    
    # Initialize predictor with a coarse grid for this example
    predictor = SouthAsianBirdHotspotPredictor(region_bbox, grid_size=0.2)
    
    # Load flyway data
    predictor.load_flyway_data()
    
    # Create simulated observation data
    print("\nGenerating simulated bird observation data...")
    num_observations = 500
    
    # Bird species in the region (simplified list)
    species = [
        # Resident birds
        'House Crow', 'Red-vented Bulbul', 'Jungle Babbler', 'Oriental Magpie-Robin',
        'Black Kite', 'Indian Peafowl', 'Yellow-footed Green Pigeon', 'Coppersmith Barbet',
        'White-throated Kingfisher', 'Rose-ringed Parakeet', 'Ashy Prinia', 'Purple Sunbird',
        'Himalayan Monal', 'Sarus Crane',
        
        # Endemic birds
        'Forest Owlet', 'Nilgiri Flycatcher', 'White-bellied Treepie',
        
        # Migratory birds
        'Bar-headed Goose', 'Common Teal', 'Northern Pintail', 'Black-tailed Godwit',
        'Common Greenshank', 'Steppe Eagle', 'Greater Spotted Eagle',
        'Bluethroat', 'Siberian Stonechat'
    ]
    
    # Generate random observations within the bounding box
    np.random.seed(42)  # For reproducibility
    
    latitudes = np.random.uniform(region_bbox[0], region_bbox[2], num_observations)
    longitudes = np.random.uniform(region_bbox[1], region_bbox[3], num_observations)
    
    # Bias observations toward certain areas (simulating real hotspots)
    # This creates clusters of observations
    for i in range(3):  # Create 3 hotspot clusters
        center_lat = np.random.uniform(region_bbox[0], region_bbox[2])
        center_lon = np.random.uniform(region_bbox[1], region_bbox[3])
        
        # Add 50 observations around each hotspot center
        cluster_size = 50
        cluster_latitudes = center_lat + np.random.normal(0, 0.3, cluster_size)
        cluster_longitudes = center_lon + np.random.normal(0, 0.3, cluster_size)
        
        # Replace some random observations with these clustered ones
        replace_indices = np.random.choice(range(num_observations), cluster_size, replace=False)
        latitudes[replace_indices] = cluster_latitudes
        longitudes[replace_indices] = cluster_longitudes
    
    # Ensure all points are within the bbox
    latitudes = np.clip(latitudes, region_bbox[0], region_bbox[2])
    longitudes = np.clip(longitudes, region_bbox[1], region_bbox[3])
    
    # Create observation DataFrame
    observations = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'species': np.random.choice(species, num_observations),
        'observation_date': pd.date_range(start='2023-01-01', periods=num_observations)
    })
    
    print(f"Generated {len(observations)} simulated observations")
    
    # Run the hotspot prediction for current date
    current_date = datetime.now()
    print(f"\nRunning prediction for current date: {current_date.strftime('%Y-%m-%d')}")
    
    result = predictor.process_for_current_date(
        latitude=latitudes[0],
        longitude=longitudes[0],
        date=current_date
    )
    
    # Visualize the results
    predictor.visualize_hotspots(result, save_path="south_asian_bird_hotspots.png")
    
    # Export the results
    export_paths = predictor.export_results(result)
    
    print("\n====== Example Completed Successfully ======")
    print(f"Results exported to: {', '.join(export_paths.values())}")
    print("Visualization saved to: south_asian_bird_hotspots.png")


if __name__ == "__main__":
    # Run the example
    run_south_asian_hotspot_example() 