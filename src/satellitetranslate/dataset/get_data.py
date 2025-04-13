import ee
import os
import datetime
import time
import numpy as np
from datetime import timedelta
import requests
import json
from concurrent.futures import ThreadPoolExecutor

# Initialize Earth Engine
ee.Authenticate()  # You'll need to authenticate first
ee.Initialize(project ="ee-get-landsat-data")

# Define sites - Name and coordinates
SITES = [
    # North America
    {"name": "san_francisco", "lat": 37.7749, "lon": -122.4194, "description": "Urban + Water"},
    {"name": "iowa", "lat": 41.8780, "lon": -93.0977, "description": "Agriculture"},
    {"name": "yellowstone", "lat": 44.4280, "lon": -110.5885, "description": "National Park"},
    {"name": "everglades", "lat": 25.2866, "lon": -80.8987, "description": "Wetlands"},
    {"name": "grand_canyon", "lat": 36.0544, "lon": -112.2583, "description": "Canyon/Desert"},
    
    # South America
    {"name": "amazon", "lat": -3.4653, "lon": -62.2159, "description": "Tropical Forest"},
    {"name": "atacama", "lat": -24.5000, "lon": -69.2500, "description": "Desert"},
    {"name": "patagonia", "lat": -49.3304, "lon": -72.8864, "description": "Glaciers/Mountains"},
    
    # Europe
    {"name": "swiss_alps", "lat": 46.8182, "lon": 8.2275, "description": "Mountains"},
    {"name": "netherlands", "lat": 52.1326, "lon": 5.2913, "description": "Agricultural/Flatlands"},
    {"name": "venice", "lat": 45.4408, "lon": 12.3155, "description": "Urban/Water"},
    
    # Africa
    {"name": "sahel", "lat": 13.5137, "lon": 2.1098, "description": "Desert/Savanna Transition"},
    {"name": "nile_delta", "lat": 30.8358, "lon": 31.0234, "description": "River Delta/Agriculture"},
    {"name": "kilimanjaro", "lat": -3.0674, "lon": 37.3556, "description": "Mountain/Forest"},
    
    # Asia
    {"name": "dubai", "lat": 25.2048, "lon": 55.2708, "description": "Urban + Desert"},
    {"name": "ganges_delta", "lat": 22.7749, "lon": 89.5565, "description": "River Delta/Dense Population"},
    {"name": "borneo", "lat": 0.9619, "lon": 114.5548, "description": "Tropical Rainforest"},
    
    # Australia/Oceania
    {"name": "great_barrier", "lat": -18.2871, "lon": 147.6992, "description": "Coral Reef/Ocean"},
    {"name": "uluru", "lat": -25.3444, "lon": 131.0369, "description": "Arid Interior"},
    {"name": "new_zealand", "lat": -43.5945, "lon": 170.2386, "description": "Mountains/Glaciers"}
]

# Time range
START_DATE = '2021-01-01'
END_DATE = '2023-12-31'

# Image parameters
LANDSAT_PATCH_SIZE = 128  # Size in pixels for Landsat (30m resolution)
SENTINEL_PATCH_SIZE = 384  # Size in pixels for Sentinel (10m resolution) - 3x larger to cover same area
MAX_CLOUD_COVER = 20  # Maximum cloud cover percentage
MAX_DAYS_DIFFERENCE = 4  # Maximum days between Landsat and Sentinel acquisitions
OUTPUT_DIR = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define bands and their names
LANDSAT_BANDS = {
    'SR_B2': 'blue',
    'SR_B3': 'green',
    'SR_B4': 'red',
    'SR_B5': 'nir',
    'SR_B6': 'swir1',
    'SR_B7': 'swir2'
}

SENTINEL_BANDS = {
    'B2': 'blue',
    'B3': 'green',
    'B4': 'red',
    'B8': 'nir',
    'B11': 'swir1',
    'B12': 'swir2'
}

# Define functions
def get_landsat_collection(start_date, end_date, cloud_cover):
    """Get Landsat 8 collection with filtering."""
    return (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover)))

def get_sentinel_collection(start_date, end_date, cloud_cover):
    """Get Sentinel-2 collection with filtering."""
    return (ee.ImageCollection('COPERNICUS/S2_SR')
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)))

def find_matching_pairs(site, max_days_diff=MAX_DAYS_DIFFERENCE):
    """Find temporally aligned Landsat 8 and Sentinel-2 image pairs."""
    point = ee.Geometry.Point([site['lon'], site['lat']])
    
    # Get collections filtered to this point
    landsat = (get_landsat_collection(START_DATE, END_DATE, MAX_CLOUD_COVER)
              .filterBounds(point))
    sentinel = (get_sentinel_collection(START_DATE, END_DATE, MAX_CLOUD_COVER)
               .filterBounds(point))
    
    # Get a list of dates for both collections
    landsat_dates = landsat.aggregate_array('system:time_start').getInfo()
    sentinel_dates = sentinel.aggregate_array('system:time_start').getInfo()
    
    # Convert to datetime for easier comparison
    landsat_datetimes = [datetime.datetime.fromtimestamp(d/1000) for d in landsat_dates]
    sentinel_datetimes = [datetime.datetime.fromtimestamp(d/1000) for d in sentinel_dates]
    
    # Find matching pairs within max_days_diff
    pairs = []
    for i, l_date in enumerate(landsat_datetimes):
        for j, s_date in enumerate(sentinel_datetimes):
            diff = abs((l_date - s_date).total_seconds() / (60*60*24))
            if diff <= max_days_diff:
                pairs.append({
                    'landsat_index': i,
                    'sentinel_index': j,
                    'landsat_date': l_date,
                    'sentinel_date': s_date,
                    'diff_days': diff
                })
    
    # Sort by date difference
    pairs.sort(key=lambda x: x['diff_days'])
    
    # Get the actual images for each pair
    result = []
    landsat_list = landsat.toList(landsat.size())
    sentinel_list = sentinel.toList(sentinel.size())
    
    for pair in pairs:
        try:
            l_img = ee.Image(landsat_list.get(pair['landsat_index']))
            s_img = ee.Image(sentinel_list.get(pair['sentinel_index']))
            
            result.append({
                'landsat': l_img,
                'sentinel': s_img,
                'landsat_date': pair['landsat_date'],
                'sentinel_date': pair['sentinel_date'],
                'diff_days': pair['diff_days']
            })
        except Exception as e:
            print(f"Error getting image pair: {e}")
    
    return result

def extract_date_string(datetime_obj):
    """Convert datetime to YYYY-MM-DD string."""
    return datetime_obj.strftime('%Y-%m-%d')

def download_single_band(image, band_name, geometry, scale, filename, patch_size):
    """Download a single band of a GEE image to a local file."""
    # Skip if file already exists
    if os.path.exists(filename):
        print(f"File already exists, skipping: {filename}")
        return True
    
    # Select just this band
    single_band_image = image.select([band_name])
    
    url = single_band_image.getDownloadURL({
        'region': geometry,
        'dimensions': f'{patch_size}x{patch_size}',
        'format': 'GEO_TIFF',
        'bands': [band_name]
    })
    
    # Download the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded: {filename}")
        return True
    else:
        print(f"Failed to download {filename}: {response.status_code}")
        return False

def process_site(site):
    """Process a single site to find and download image pairs."""
    site_name = site['name']
    print(f"Processing site: {site_name}")
    
    # Create site directory
    site_dir = os.path.join(OUTPUT_DIR, site_name)
    os.makedirs(site_dir, exist_ok=True)
    
    # Find matching image pairs
    point = ee.Geometry.Point([site['lon'], site['lat']])
    # Use 30m resolution (Landsat) for buffer to ensure both images cover the same area
    buffer_distance = LANDSAT_PATCH_SIZE * 30  # Buffer in meters (30m Landsat resolution)
    region = point.buffer(buffer_distance).bounds()
    
    pairs = find_matching_pairs(site)
    print(f"Found {len(pairs)} matching pairs for {site_name}")
    
    # Counter for tracking actual downloads
    pairs_downloaded = 0
    pairs_skipped = 0
    
    # Download each pair
    for i, pair in enumerate(pairs):
        landsat_date = extract_date_string(pair['landsat_date'])
        sentinel_date = extract_date_string(pair['sentinel_date'])
        
        # Create date directory using landsat date (reference date)
        date_dir = os.path.join(site_dir, landsat_date)
        landsat_dir = os.path.join(date_dir, 'landsat8')
        sentinel_dir = os.path.join(date_dir, 'sentinel2')
        
        os.makedirs(landsat_dir, exist_ok=True)
        os.makedirs(sentinel_dir, exist_ok=True)
        
        # Check if all bands already exist
        all_landsat_exist = all(os.path.exists(os.path.join(landsat_dir, f'landsat8_{site_name}_{landsat_date}_{band_desc}.tif')) 
                              for band_desc in LANDSAT_BANDS.values())
        all_sentinel_exist = all(os.path.exists(os.path.join(sentinel_dir, f'sentinel2_{site_name}_{sentinel_date}_{band_desc}.tif')) 
                               for band_desc in SENTINEL_BANDS.values())
        
        if all_landsat_exist and all_sentinel_exist:
            print(f"Pair {i+1}/{len(pairs)} for {site_name} already downloaded, skipping")
            pairs_skipped += 1
            continue
            
        # Save metadata about the pair
        metadata_file = os.path.join(date_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            with open(metadata_file, 'w') as f:
                json.dump({
                    'landsat_date': landsat_date,
                    'sentinel_date': sentinel_date,
                    'diff_days': pair['diff_days'],
                    'site': site,
                    'landsat_resolution': '30m',
                    'sentinel_resolution': '10m',
                    'landsat_patch_size': LANDSAT_PATCH_SIZE,
                    'sentinel_patch_size': SENTINEL_PATCH_SIZE,
                    'landsat_bands': list(LANDSAT_BANDS.keys()),
                    'sentinel_bands': list(SENTINEL_BANDS.keys())
                }, f, indent=2, default=str)
        
        # Download each Landsat band
        landsat_success = True
        for band_code, band_desc in LANDSAT_BANDS.items():
            landsat_file = os.path.join(landsat_dir, f'landsat8_{site_name}_{landsat_date}_{band_desc}.tif')
            
            if not os.path.exists(landsat_file):
                band_success = download_single_band(
                    pair['landsat'], band_code, region, 30, landsat_file, LANDSAT_PATCH_SIZE
                )
                if not band_success:
                    landsat_success = False
                    break
                # Small delay between band downloads
                time.sleep(0.5)
        
        # Download each Sentinel band if Landsat was successful
        sentinel_success = True
        if landsat_success:
            for band_code, band_desc in SENTINEL_BANDS.items():
                sentinel_file = os.path.join(sentinel_dir, f'sentinel2_{site_name}_{sentinel_date}_{band_desc}.tif')
                
                if not os.path.exists(sentinel_file):
                    band_success = download_single_band(
                        pair['sentinel'], band_code, region, 10, sentinel_file, SENTINEL_PATCH_SIZE
                    )
                    if not band_success:
                        sentinel_success = False
                        break
                    # Small delay between band downloads
                    time.sleep(0.5)
            
            if sentinel_success:
                pairs_downloaded += 1
                print(f"Downloaded pair {i+1}/{len(pairs)} for {site_name}")
        
        # Add a delay to avoid rate limiting
        time.sleep(1)
    
    print(f"Site {site_name} summary: {pairs_downloaded} pairs downloaded, {pairs_skipped} pairs skipped")

# Process all sites
def main():
    print(f"Starting SatelliteTranslate data collection (separate bands)")
    print(f"Time range: {START_DATE} to {END_DATE}")
    print(f"Max cloud cover: {MAX_CLOUD_COVER}%")
    print(f"Max days between acquisitions: {MAX_DAYS_DIFFERENCE}")
    print(f"Landsat patch size: {LANDSAT_PATCH_SIZE}x{LANDSAT_PATCH_SIZE} pixels (30m resolution)")
    print(f"Sentinel patch size: {SENTINEL_PATCH_SIZE}x{SENTINEL_PATCH_SIZE} pixels (10m resolution)")
    print(f"Sites to process: {len(SITES)}")
    print(f"Landsat bands: {', '.join([f'{k} ({v})' for k, v in LANDSAT_BANDS.items()])}")
    print(f"Sentinel bands: {', '.join([f'{k} ({v})' for k, v in SENTINEL_BANDS.items()])}")
    
    # Create a summary file
    with open(os.path.join(OUTPUT_DIR, 'dataset_info.txt'), 'w') as f:
        f.write(f"SatelliteTranslate Dataset (separate bands)\n")
        f.write(f"Created: {datetime.datetime.now()}\n")
        f.write(f"Time range: {START_DATE} to {END_DATE}\n")
        f.write(f"Resolution transformation: Landsat 8 (30m) to Sentinel-2 (10m)\n")
        f.write(f"Patch sizes: Landsat {LANDSAT_PATCH_SIZE}x{LANDSAT_PATCH_SIZE}, Sentinel {SENTINEL_PATCH_SIZE}x{SENTINEL_PATCH_SIZE}\n")
        f.write(f"Landsat bands: {', '.join([f'{k} ({v})' for k, v in LANDSAT_BANDS.items()])}\n")
        f.write(f"Sentinel bands: {', '.join([f'{k} ({v})' for k, v in SENTINEL_BANDS.items()])}\n")
        f.write(f"Sites:\n")
        for site in SITES:
            f.write(f"  - {site['name']}: {site['description']} ({site['lat']}, {site['lon']})\n")
    
    # Process sites with multiple workers
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_site, SITES)
    
    print(f"Data collection complete. Check the {OUTPUT_DIR} directory for results.")

if __name__ == "__main__":
    main()