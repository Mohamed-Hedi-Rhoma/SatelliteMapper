import os
import glob
import json
import shutil
import numpy as np
import re
from datetime import datetime, timedelta

# Define directories
SENTINEL_DATA_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data_sentinel2"
LANDSAT_DATA_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/landsat_data"
OUTPUT_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data"

# Maximum time difference in days
MAX_TIME_DIFF = 5

def parse_date(date_str):
    """
    Parse date string in format 'YYYY-MM-DD' to datetime object.
    If the date has a suffix like '_SZA', it will be removed.
    """
    # Remove any suffix after the date (like _SZA)
    if '_' in date_str:
        date_str = date_str.split('_')[0]
    return datetime.strptime(date_str, "%Y-%m-%d")

def find_closest_landsat_date(sentinel_date, landsat_dates):
    """
    Find the closest Landsat date to the given Sentinel date within MAX_TIME_DIFF days.
    
    Args:
        sentinel_date (str): Sentinel-2 acquisition date
        landsat_dates (list): List of Landsat 8 acquisition dates
    
    Returns:
        str or None: The closest Landsat date if found, None otherwise
    """
    # Convert to datetime for comparison
    sentinel_dt = parse_date(sentinel_date)
    
    # Calculate time differences
    time_diffs = [(abs((parse_date(d) - sentinel_dt).days), d) for d in landsat_dates]
    
    # Find the closest date within MAX_TIME_DIFF days
    valid_diffs = [item for item in time_diffs if item[0] <= MAX_TIME_DIFF]
    
    if valid_diffs:
        # Return the date with minimum difference
        valid_diffs.sort(key=lambda x: x[0])  # Sort by difference
        return valid_diffs[0][1]  # Return the date
    
    return None

def extract_landsat_angles(site_path, date):
    """
    Extract angle means from Landsat 8 metadata JSON.
    The Landsat format uses keys like "2018-01-10_SZA" in the angles subdictionaries.
    
    Args:
        site_path (str): Path to the Landsat site directory
        date (str): Acquisition date
    
    Returns:
        dict: Dictionary with angle means
    """
    metadata_path = os.path.join(site_path, "site_metadata.json")
    
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found at {metadata_path}")
        return {}
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # For Landsat, the angles are stored in a specific structure
        normalized_angles = {
            "SZA": None,
            "VZA": None,
            "SAA": None,
            "VAA": None
        }
        
        if "angles" in metadata:
            angle_types = ["SZA", "VZA", "SAA", "VAA"]
            for angle_type in angle_types:
                if angle_type in metadata["angles"]:
                    # Look for date_angle format keys
                    date_key = f"{date}_{angle_type}"
                    if date_key in metadata["angles"][angle_type]:
                        # Divide by 100 to get the real values
                        normalized_angles[angle_type] = metadata["angles"][angle_type][date_key] / 100.0
        
        return normalized_angles
        
    except Exception as e:
        print(f"Error extracting Landsat angles: {e}")
        return {}

def extract_sentinel_angles(site_path, date):
    """
    Extract angle means from Sentinel-2 metadata JSON and calculate average incidence angles.
    
    Args:
        site_path (str): Path to the Sentinel site directory
        date (str): Acquisition date
    
    Returns:
        dict: Dictionary with angle means in the same format as Landsat
    """
    metadata_path = os.path.join(site_path, "site_metadata.json")
    
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found at {metadata_path}")
        return {}
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Initialize normalized angles dictionary
        normalized_angles = {
            "SZA": None,
            "VZA": None,
            "SAA": None,
            "VAA": None
        }
        
        # For Sentinel-2, check in angles_by_date
        if "angles_by_date" in metadata and date in metadata["angles_by_date"]:
            angles = metadata["angles_by_date"][date]
            
            # Map Sentinel-2 solar angles to Landsat format
            if "MEAN_SOLAR_ZENITH_ANGLE" in angles:
                normalized_angles["SZA"] = angles["MEAN_SOLAR_ZENITH_ANGLE"]
                
            if "MEAN_SOLAR_AZIMUTH_ANGLE" in angles:
                normalized_angles["SAA"] = angles["MEAN_SOLAR_AZIMUTH_ANGLE"]
            
            # Calculate mean incidence angles across all bands
            zenith_values = []
            azimuth_values = []
            
            for key, value in angles.items():
                if "MEAN_INCIDENCE_ZENITH_ANGLE" in key:
                    zenith_values.append(value)
                elif "MEAN_INCIDENCE_AZIMUTH_ANGLE" in key:
                    azimuth_values.append(value)
            
            if zenith_values:
                normalized_angles["VZA"] = sum(zenith_values) / len(zenith_values)
                
            if azimuth_values:
                normalized_angles["VAA"] = sum(azimuth_values) / len(azimuth_values)
            
        return normalized_angles
            
    except Exception as e:
        print(f"Error extracting Sentinel angles: {e}")
        return {}

def get_all_dates_from_dirs(directory):
    """
    Get all date folders from a directory.
    
    Args:
        directory (str): Path to search for date folders
        
    Returns:
        list: List of date folder names (YYYY-MM-DD format)
    """
    # Get all directories that look like dates (YYYY-MM-DD)
    all_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    date_dirs = [d for d in all_dirs if re.match(r'^\d{4}-\d{2}-\d{2}$', d)]
    return date_dirs

def create_image_pairs():
    """
    Create pairs of Sentinel-2 and Landsat 8 images based on acquisition time.
    """
    print("Starting to create image pairs...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all site directories for Sentinel-2
    sentinel_site_dirs = [d for d in glob.glob(os.path.join(SENTINEL_DATA_DIR, "*")) 
                         if os.path.isdir(d) and not d.endswith("__pycache__")]
    
    # Counter for successful pairs
    pair_count = 0
    
    # Process each Sentinel-2 site
    for sentinel_site_dir in sentinel_site_dirs:
        site_name = os.path.basename(sentinel_site_dir)
        print(f"Processing site: {site_name}")
        
        # Find corresponding Landsat site
        landsat_site_dir = os.path.join(LANDSAT_DATA_DIR, site_name)
        if not os.path.exists(landsat_site_dir):
            print(f"  Landsat site directory not found for {site_name}. Skipping.")
            continue
        
        # Get all date directories for this Sentinel site
        sentinel_dates = get_all_dates_from_dirs(sentinel_site_dir)
        
        # Get all date directories for this Landsat site
        landsat_dates = get_all_dates_from_dirs(landsat_site_dir)
        
        print(f"  Found {len(sentinel_dates)} Sentinel dates and {len(landsat_dates)} Landsat dates")
        
        # For each Sentinel date, find the closest Landsat date
        for sentinel_date in sentinel_dates:
            closest_landsat_date = find_closest_landsat_date(sentinel_date, landsat_dates)
            
            if closest_landsat_date:
                # Calculate days difference
                days_diff = abs((parse_date(closest_landsat_date) - parse_date(sentinel_date)).days)
                print(f"  Pair found: Sentinel {sentinel_date} - Landsat {closest_landsat_date} ({days_diff} days apart)")
                
                # Create output directories with site name included
                site_dir = os.path.join(OUTPUT_DIR, site_name)
                pair_dir = os.path.join(site_dir, f"{sentinel_date}")
                sentinel_output_dir = os.path.join(pair_dir, "sentinel2")
                landsat_output_dir = os.path.join(pair_dir, "landsat8")
                
                os.makedirs(site_dir, exist_ok=True)
                os.makedirs(pair_dir, exist_ok=True)
                os.makedirs(sentinel_output_dir, exist_ok=True)
                os.makedirs(landsat_output_dir, exist_ok=True)
                
                # Copy Sentinel-2 files
                sentinel_date_dir = os.path.join(sentinel_site_dir, sentinel_date)
                sentinel_files = glob.glob(os.path.join(sentinel_date_dir, "*.tif"))
                for file in sentinel_files:
                    shutil.copy2(file, sentinel_output_dir)
                
                # Extract and save Sentinel-2 angles
                sentinel_angles = extract_sentinel_angles(sentinel_site_dir, sentinel_date)
                if sentinel_angles:
                    with open(os.path.join(sentinel_output_dir, "angles.json"), 'w') as f:
                        json.dump(sentinel_angles, f, indent=4)
                
                # Copy Landsat 8 files
                landsat_date_dir = os.path.join(landsat_site_dir, closest_landsat_date)
                landsat_files = glob.glob(os.path.join(landsat_date_dir, "*.tif"))
                for file in landsat_files:
                    shutil.copy2(file, landsat_output_dir)
                
                # Extract and save Landsat 8 angles
                landsat_angles = extract_landsat_angles(landsat_site_dir, closest_landsat_date)
                if landsat_angles:
                    with open(os.path.join(landsat_output_dir, "angles.json"), 'w') as f:
                        json.dump(landsat_angles, f, indent=4)
                
                pair_count += 1
            else:
                print(f"  No matching Landsat date found for Sentinel date {sentinel_date} within {MAX_TIME_DIFF} days")
    
    print(f"Image pair creation completed. Created {pair_count} pairs.")

if __name__ == "__main__":
    create_image_pairs()