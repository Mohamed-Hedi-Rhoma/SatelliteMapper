import ee
import os
import datetime
import time
import numpy as np
from datetime import timedelta
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import rasterio
import shutil
import tempfile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Initialize Earth Engine
ee.Authenticate()  # You'll need to authenticate first
ee.Initialize(project="ee-get-landsat-data")

# Define sites - Name and coordinates
SITES = [
    # North America
    {"name": "san_francisco", "lat": 37.7749, "lon": -122.4194, "description": "Urban + Water"},
    {"name": "iowa", "lat": 41.8780, "lon": -93.0977, "description": "Agriculture"},
    {"name": "yellowstone", "lat": 44.4280, "lon": -110.5885, "description": "National Park"},
    {"name": "everglades", "lat": 25.2866, "lon": -80.8987, "description": "Wetlands"},
    {"name": "grand_canyon", "lat": 36.0544, "lon": -112.2583, "description": "Canyon/Desert"},
    {"name": "chesapeake_bay", "lat": 37.8395, "lon": -76.0193, "description": "Estuary/Coastal"},
    {"name": "death_valley", "lat": 36.5323, "lon": -116.9325, "description": "Desert/Basin"},
    {"name": "great_lakes", "lat": 45.0000, "lon": -83.0000, "description": "Freshwater Lakes"},
    {"name": "vancouver_island", "lat": 49.6819, "lon": -125.4514, "description": "Temperate Rainforest"},
    {"name": "quebec_taiga", "lat": 52.9399, "lon": -73.5491, "description": "Boreal Forest"},
    {"name": "mexico_city", "lat": 19.4326, "lon": -99.1332, "description": "Urban/High Altitude"},
    {"name": "sonoran_desert", "lat": 32.2528, "lon": -112.8714, "description": "Desert Ecosystem"},
    
    # South America
    {"name": "amazon", "lat": -3.4653, "lon": -62.2159, "description": "Tropical Forest"},
    {"name": "atacama", "lat": -24.5000, "lon": -69.2500, "description": "Desert"},
    {"name": "patagonia", "lat": -49.3304, "lon": -72.8864, "description": "Glaciers/Mountains"},
    {"name": "pantanal", "lat": -17.6117, "lon": -57.4286, "description": "Wetlands/Savanna"},
    {"name": "buenos_aires", "lat": -34.6037, "lon": -58.3816, "description": "Urban/Pampas"},
    {"name": "galapagos", "lat": -0.7393, "lon": -90.3305, "description": "Volcanic Islands"},
    
    # Europe
    {"name": "swiss_alps", "lat": 46.8182, "lon": 8.2275, "description": "Mountains"},
    {"name": "netherlands", "lat": 52.1326, "lon": 5.2913, "description": "Agricultural/Flatlands"},
    {"name": "venice", "lat": 45.4408, "lon": 12.3155, "description": "Urban/Water"},
    {"name": "pyrenees", "lat": 42.6582, "lon": 1.0057, "description": "Mountain Range"},
    {"name": "danube_delta", "lat": 45.0000, "lon": 29.0000, "description": "River Delta/Wetlands"},
    {"name": "iceland", "lat": 64.9631, "lon": -19.0208, "description": "Volcanic/Glaciers"},
    {"name": "london", "lat": 51.5074, "lon": -0.1278, "description": "Urban/River"},
    {"name": "scandinavian_tundra", "lat": 68.3700, "lon": 18.8300, "description": "Tundra/Arctic"},
    
    # Africa
    {"name": "sahel", "lat": 13.5137, "lon": 2.1098, "description": "Desert/Savanna Transition"},
    {"name": "nile_delta", "lat": 30.8358, "lon": 31.0234, "description": "River Delta/Agriculture"},
    {"name": "kilimanjaro", "lat": -3.0674, "lon": 37.3556, "description": "Mountain/Forest"},
    {"name": "serengeti", "lat": -2.3333, "lon": 34.8333, "description": "Savanna/Grassland"},
    {"name": "congo_basin", "lat": -0.7832, "lon": 23.6558, "description": "Tropical Rainforest"},
    {"name": "sahara", "lat": 23.4162, "lon": 25.6628, "description": "Desert"},
    {"name": "lake_victoria", "lat": -1.2650, "lon": 33.2418, "description": "Freshwater Lake"},
    {"name": "okavango_delta", "lat": -19.2330, "lon": 22.8755, "description": "Inland Delta"},
    
    # Asia
    {"name": "dubai", "lat": 25.2048, "lon": 55.2708, "description": "Urban + Desert"},
    {"name": "ganges_delta", "lat": 22.7749, "lon": 89.5565, "description": "River Delta/Dense Population"},
    {"name": "borneo", "lat": 0.9619, "lon": 114.5548, "description": "Tropical Rainforest"},
    {"name": "aral_sea", "lat": 45.0000, "lon": 60.0000, "description": "Shrinking Lake/Desertification"},
    {"name": "siberian_taiga", "lat": 60.0000, "lon": 100.0000, "description": "Boreal Forest"},
    {"name": "himalaya", "lat": 28.2460, "lon": 85.3131, "description": "High Mountains"},
    {"name": "tokyo", "lat": 35.6762, "lon": 139.6503, "description": "Urban/Coastal"},
    {"name": "mekong_delta", "lat": 10.0341, "lon": 105.7841, "description": "River Delta/Agriculture"},
    {"name": "gobi_desert", "lat": 42.5676, "lon": 103.9552, "description": "Cold Desert"},
    {"name": "baikal", "lat": 53.5000, "lon": 108.0000, "description": "Deep Lake/Forest"},
    
    # Australia/Oceania
    {"name": "great_barrier", "lat": -18.2871, "lon": 147.6992, "description": "Coral Reef/Ocean"},
    {"name": "uluru", "lat": -25.3444, "lon": 131.0369, "description": "Arid Interior"},
    {"name": "new_zealand", "lat": -43.5945, "lon": 170.2386, "description": "Mountains/Glaciers"},
    {"name": "sydney", "lat": -33.8688, "lon": 151.2093, "description": "Urban/Coastal"},
    {"name": "great_victoria_desert", "lat": -29.0000, "lon": 127.0000, "description": "Desert"},
    {"name": "tasmania", "lat": -42.0000, "lon": 146.0000, "description": "Temperate Forest Island"},
    {"name": "papua_highlands", "lat": -4.0000, "lon": 141.0000, "description": "Tropical Mountain Forest"},
    {"name": "coral_sea", "lat": -18.0000, "lon": 157.0000, "description": "Open Ocean/Reefs"}
]

# Time range
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# Image parameters
LANDSAT_PATCH_SIZE = 128  # Size in pixels for Landsat (30m resolution)
SENTINEL_PATCH_SIZE = 384  # Size in pixels for Sentinel (10m resolution) - 3x larger to cover same area
MAX_CLOUD_COVER = 10  # Maximum cloud cover percentage
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
    'SR_B7': 'swir2',
    'QA_PIXEL': 'qa_pixel'  
}

SENTINEL_BANDS = {
    'B2': 'blue',
    'B3': 'green',
    'B4': 'red',
    'B8': 'nir',
    'B11': 'swir1',
    'B12': 'swir2',
    'QA60': 'qa_pixel'  
}

# Define angle bands for mean calculation
LANDSAT_ANGLES = {
    'SAA': 'solar_azimuth',
    'SZA': 'solar_zenith',
    'VAA': 'view_azimuth',
    'VZA': 'view_zenith'
}

# Sentinel-2 angle properties
SENTINEL_ANGLE_PROPERTIES = {
    'MEAN_SOLAR_AZIMUTH_ANGLE': 'solar_azimuth',
    'MEAN_SOLAR_ZENITH_ANGLE': 'solar_zenith',
    'MEAN_INCIDENCE_AZIMUTH_ANGLE_B2': 'view_azimuth',
    'MEAN_INCIDENCE_ZENITH_ANGLE_B2': 'view_zenith'
}

# Create optimized session for faster downloads
def create_optimized_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # Maximum number of retries
        backoff_factor=0.5,  # Backoff factor for retry delay
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET"]  # Only retry on GET
    )
    adapter = HTTPAdapter(
        pool_connections=25,  # Increase connection pool size
        pool_maxsize=25,      # Increase max connections
        max_retries=retry_strategy
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Create a global session for reuse
session = create_optimized_session()

# Define functions
def get_landsat_collection(start_date, end_date, cloud_cover):
    """Get Landsat 8 Level 2 collection with filtering."""
    return (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover)))

def get_landsat_angles_collection(start_date, end_date, cloud_cover):
    """Get Landsat 8 Level 1 collection for angle data."""
    return (ee.ImageCollection('LANDSAT/LC08/C02/T1')
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover)))

def get_sentinel_collection(start_date, end_date, cloud_cover):
    """Get Sentinel-2 collection with filtering."""
    return (ee.ImageCollection('COPERNICUS/S2_SR')
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)))

def extract_sentinel_angle_means(sentinel_image):
    """Extract mean angle values from Sentinel-2 metadata."""
    angle_means = {}
    
    for prop_name, angle_name in SENTINEL_ANGLE_PROPERTIES.items():
        try:
            angle_value = sentinel_image.get(prop_name).getInfo()
            angle_means[angle_name] = angle_value
        except Exception as e:
            print(f"Warning: Could not extract {prop_name} from Sentinel-2 metadata: {e}")
            angle_means[angle_name] = None
            
    return angle_means

def find_matching_pairs(site, max_days_diff=MAX_DAYS_DIFFERENCE):
    """Find temporally aligned Landsat 8 and Sentinel-2 image pairs."""
    point = ee.Geometry.Point([site['lon'], site['lat']])
    
    # Get collections filtered to this point
    landsat = (get_landsat_collection(START_DATE, END_DATE, MAX_CLOUD_COVER)
              .filterBounds(point))
    landsat_angles = (get_landsat_angles_collection(START_DATE, END_DATE, MAX_CLOUD_COVER)
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
    landsat_angles_list = landsat_angles.toList(landsat_angles.size())
    
    for pair in pairs:
        try:
            l_img = ee.Image(landsat_list.get(pair['landsat_index']))
            s_img = ee.Image(sentinel_list.get(pair['sentinel_index']))
            
            # Find matching Landsat L1 image for angles
            l_time = l_img.get('system:time_start')
            l_angles_imgs = landsat_angles.filter(ee.Filter.equals('system:time_start', l_time))
            
            # If no exact match is found, use the closest in time
            if l_angles_imgs.size().getInfo() == 0:
                l_angles_imgs = landsat_angles.filter(
                    ee.Filter.calendarRange(
                        pair['landsat_date'].year, pair['landsat_date'].year, 'year'
                    ).And(
                        ee.Filter.calendarRange(
                            pair['landsat_date'].month, pair['landsat_date'].month, 'month'
                        ).And(
                            ee.Filter.calendarRange(
                                pair['landsat_date'].day, pair['landsat_date'].day, 'day'
                            )
                        )
                    )
                )
            
            # If we still don't have an angle image, skip this pair
            if l_angles_imgs.size().getInfo() == 0:
                print(f"No matching Landsat angle data found for {pair['landsat_date']}, skipping pair")
                continue
                
            l_angles_img = ee.Image(l_angles_imgs.first())
            
            # Extract Sentinel-2 angle means directly from metadata
            s_angle_means = extract_sentinel_angle_means(s_img)
            
            result.append({
                'landsat': l_img,
                'landsat_angles': l_angles_img,
                'sentinel': s_img,
                'landsat_date': pair['landsat_date'],
                'sentinel_date': pair['sentinel_date'],
                'sentinel_angle_means': s_angle_means,
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
    
    # Download the image with retry and stream
    try:
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # Use streaming to avoid memory issues with large files
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                f.write(chunk)
                
        print(f"Successfully downloaded: {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {str(e)}")
        return False

def calculate_tif_mean(tif_file):
    """Calculate the mean value of a GeoTIFF file."""
    with rasterio.open(tif_file) as src:
        data = src.read(1)
        # Filter out no data values (assuming they're very negative or 0)
        valid_data = data[data > -9999]
        valid_data = valid_data[valid_data != 0]
        if len(valid_data) > 0:
            return float(np.mean(valid_data))
        else:
            return None

def download_bands_parallel(image, bands_dict, geometry, scale, output_dir, prefix, site_name, date_str, patch_size, max_workers=8):
    """Download multiple bands in parallel using thread pool."""
    failed_bands = []
    
    # Create temporary directory for partial downloads to avoid corrupted files
    temp_download_dir = os.path.join(output_dir, 'temp_downloads')
    os.makedirs(temp_download_dir, exist_ok=True)
    
    def download_band_task(band_code, band_desc):
        final_filename = os.path.join(output_dir, f'{prefix}_{site_name}_{date_str}_{band_desc}.tif')
        if os.path.exists(final_filename):
            return band_code, band_desc, True
            
        # Download to temporary file first
        temp_filename = os.path.join(temp_download_dir, f'temp_{prefix}_{band_code}_{int(time.time())}.tif')
        success = download_single_band(image, band_code, geometry, scale, temp_filename, patch_size)
        
        if success:
            # Move to final location
            shutil.move(temp_filename, final_filename)
            return band_code, band_desc, True
        else:
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                pass
            return band_code, band_desc, False
    
    # Use thread pool for band downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_band = {
            executor.submit(download_band_task, band_code, band_desc): (band_code, band_desc)
            for band_code, band_desc in bands_dict.items()
        }
        
        # Process completed tasks
        for future in as_completed(future_to_band):
            band_code, band_desc, success = future.result()
            if not success:
                failed_bands.append((band_code, band_desc))
    
    # Clean up temporary directory
    try:
        shutil.rmtree(temp_download_dir)
    except Exception as e:
        print(f"Warning: Could not remove temp download directory: {e}")
    
    # Return success if all bands downloaded successfully
    return len(failed_bands) == 0

def download_and_process_landsat_angles(pair, site_name, temp_dir, region, landsat_date):
    """Download Landsat angle bands, calculate means, and return the values with parallel processing."""
    angle_means = {}
    futures = []
    
    # Process angle bands in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        for band_code, angle_name in LANDSAT_ANGLES.items():
            temp_file = os.path.join(temp_dir, f'temp_landsat_{angle_name}.tif')
            
            # Skip if already processed
            if angle_name in angle_means:
                continue
                
            # Submit download task
            def process_angle_band(band_code, angle_name, temp_file):
                success = download_single_band(
                    pair['landsat_angles'], band_code, region, 30, temp_file, LANDSAT_PATCH_SIZE
                )
                
                if success:
                    # Calculate mean
                    mean_value = calculate_tif_mean(temp_file)
                    # Divide by 100 for Landsat angles
                    if mean_value is not None:
                        mean_value = mean_value / 100.0
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                        
                    return angle_name, mean_value
                else:
                    return angle_name, None
            
            # Submit task
            future = executor.submit(process_angle_band, band_code, angle_name, temp_file)
            futures.append((future, angle_name))
            
        # Get results
        for future, angle_name in futures:
            try:
                angle_name, mean_value = future.result()
                angle_means[angle_name] = mean_value
                print(f"Calculated mean for {angle_name}: {mean_value}")
            except Exception as e:
                print(f"Error processing angle {angle_name}: {e}")
                angle_means[angle_name] = None
    
    return angle_means

def process_site(site):
    """Process a single site to find and download image pairs."""
    site_name = site['name']
    print(f"Processing site: {site_name}")
    
    # Create site directory
    site_dir = os.path.join(OUTPUT_DIR, site_name)
    os.makedirs(site_dir, exist_ok=True)
    
    # Create temporary directory for angle calculation
    temp_dir = os.path.join(site_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
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
        angles_json_exists = os.path.exists(os.path.join(date_dir, 'angles.json'))
        
        if all_landsat_exist and all_sentinel_exist and angles_json_exists:
            print(f"Pair {i+1}/{len(pairs)} for {site_name} already downloaded, skipping")
            pairs_skipped += 1
            continue
        
        # Process angles first to avoid downloading spectral data if angle processing fails
        if not angles_json_exists:
            # Get Landsat angle means
            landsat_angle_means = download_and_process_landsat_angles(
                pair, site_name, temp_dir, region, landsat_date
            )
            
            # Already have Sentinel angle means from the metadata
            sentinel_angle_means = pair['sentinel_angle_means']
            
            # Save both sets of angles to a JSON file
            angles_data = {
                'landsat': {
                    'date': landsat_date,
                    'angles': landsat_angle_means
                },
                'sentinel': {
                    'date': sentinel_date,
                    'angles': sentinel_angle_means
                }
            }
            
            with open(os.path.join(date_dir, 'angles.json'), 'w') as f:
                json.dump(angles_data, f, indent=2)
        
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
        
        # Download all Landsat bands in parallel
        if not all_landsat_exist:
            landsat_success = download_bands_parallel(
                pair['landsat'],
                LANDSAT_BANDS,
                region,
                30,
                landsat_dir,
                'landsat8',
                site_name,
                landsat_date,
                LANDSAT_PATCH_SIZE
            )
        else:
            landsat_success = True
            
        # Download all Sentinel bands in parallel if Landsat was successful
        sentinel_success = False
        if landsat_success and not all_sentinel_exist:
            sentinel_success = download_bands_parallel(
                pair['sentinel'],
                SENTINEL_BANDS,
                region,
                10,
                sentinel_dir,
                'sentinel2',
                site_name,
                sentinel_date,
                SENTINEL_PATCH_SIZE
            )
        elif landsat_success and all_sentinel_exist:
            sentinel_success = True
            
        if landsat_success and sentinel_success and not (all_landsat_exist and all_sentinel_exist):
            pairs_downloaded += 1
            print(f"Downloaded pair {i+1}/{len(pairs)} for {site_name}")
        
        # Small delay between pairs to avoid overwhelming GEE
        time.sleep(0.1)
    
    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not remove temp directory: {e}")
    
    print(f"Site {site_name} summary: {pairs_downloaded} pairs downloaded, {pairs_skipped} pairs skipped")
    return pairs_downloaded, pairs_skipped

def main():
    print(f"Starting SatelliteTranslate data collection (with mean angles)")
    print(f"Time range: {START_DATE} to {END_DATE}")
    print(f"Max cloud cover: {MAX_CLOUD_COVER}%")
    print(f"Max days between acquisitions: {MAX_DAYS_DIFFERENCE}")
    print(f"Landsat patch size: {LANDSAT_PATCH_SIZE}x{LANDSAT_PATCH_SIZE} pixels (30m resolution)")
    print(f"Sentinel patch size: {SENTINEL_PATCH_SIZE}x{SENTINEL_PATCH_SIZE} pixels (10m resolution)")
    print(f"Sites to process: {len(SITES)}")
    print(f"Landsat bands: {', '.join([f'{k} ({v})' for k, v in LANDSAT_BANDS.items()])}")
    print(f"Sentinel bands: {', '.join([f'{k} ({v})' for k, v in SENTINEL_BANDS.items()])}")
    print(f"Using mean angles for both satellites stored in angles.json")
    print(f"Using optimized parallel download with connection pooling")
    
    # Create a summary file
    with open(os.path.join(OUTPUT_DIR, 'dataset_info.txt'), 'w') as f:
        f.write(f"SatelliteTranslate Dataset (with mean angles)\n")
        f.write(f"Created: {datetime.datetime.now()}\n")
        f.write(f"Time range: {START_DATE} to {END_DATE}\n")
        f.write(f"Resolution transformation: Landsat 8 (30m) to Sentinel-2 (10m)\n")
        f.write(f"Patch sizes: Landsat {LANDSAT_PATCH_SIZE}x{LANDSAT_PATCH_SIZE}, Sentinel {SENTINEL_PATCH_SIZE}x{SENTINEL_PATCH_SIZE}\n")
        f.write(f"Landsat bands: {', '.join([f'{k} ({v})' for k, v in LANDSAT_BANDS.items()])}\n")
        f.write(f"Sentinel bands: {', '.join([f'{k} ({v})' for k, v in SENTINEL_BANDS.items()])}\n")
        f.write(f"Angle information: Scene-average values stored in angles.json\n")
        f.write(f"  - Landsat angles: {', '.join([f'{v}' for v in LANDSAT_ANGLES.values()])}\n")
        f.write(f"  - Sentinel angles: {', '.join([f'{v}' for v in SENTINEL_ANGLE_PROPERTIES.values()])}\n")
        f.write(f"Sites:\n")
        for site in SITES:
            f.write(f"  - {site['name']}: {site['description']} ({site['lat']}, {site['lon']})\n")
    
    # Process sites with thread pool
    total_pairs_downloaded = 0
    total_pairs_skipped = 0
    
    # Adjust max_workers based on your system's capabilities 
    # Usually number of CPU cores * 2 is a good starting point
    max_site_workers = 6  # Adjust based on your system
    
    with ThreadPoolExecutor(max_workers=max_site_workers) as executor:
        future_to_site = {executor.submit(process_site, site): site for site in SITES}
        
        # Process results as they complete
        for future in as_completed(future_to_site):
            site = future_to_site[future]
            try:
                pairs_downloaded, pairs_skipped = future.result()
                total_pairs_downloaded += pairs_downloaded
                total_pairs_skipped += pairs_skipped
                print(f"Completed site {site['name']}")
            except Exception as e:
                print(f"Error processing site {site['name']}: {e}")
    
    print(f"Data collection complete.")
    print(f"Total pairs downloaded: {total_pairs_downloaded}")
    print(f"Total pairs skipped: {total_pairs_skipped}")
    print(f"Check the {OUTPUT_DIR} directory for results.")

if __name__ == "__main__":
    main()