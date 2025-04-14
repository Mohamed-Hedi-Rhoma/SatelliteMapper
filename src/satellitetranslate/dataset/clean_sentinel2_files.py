import os
import re
import datetime
from tqdm import tqdm
from collections import defaultdict

def clean_sentinel_files_by_date(base_dir):
    """
    Simple script to clean Sentinel-2 files by keeping only those closest to the Landsat date.
    The Landsat date is assumed to be the name of the directory.
    
    Args:
        base_dir (str): Base directory containing the satellite data
    """
    # Get all site folders
    site_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) 
                   and not f.startswith('.') and f != 'dataset_info.txt']
    
    # Stats counters
    total_dates_processed = 0
    total_files_kept = 0
    total_files_removed = 0
    
    for site in tqdm(site_folders, desc="Processing sites"):
        site_path = os.path.join(base_dir, site)
        
        # Get all date folders for the site (excluding _scaled folders)
        date_folders = [d for d in os.listdir(site_path) if os.path.isdir(os.path.join(site_path, d)) 
                       and not d.endswith('_scaled')]
        
        for landsat_date_str in tqdm(date_folders, desc=f"Dates in {site}", leave=False):
            date_path = os.path.join(site_path, landsat_date_str)
            sentinel_dir = os.path.join(date_path, 'sentinel2')
            
            # Skip if sentinel directory doesn't exist
            if not os.path.exists(sentinel_dir):
                continue
                
            # The landsat date is the folder name
            try:
                landsat_date = datetime.datetime.strptime(landsat_date_str, '%Y-%m-%d')
            except ValueError:
                print(f"Warning: Folder name {landsat_date_str} is not a valid date format, skipping")
                continue
                
            total_dates_processed += 1
            
            # Get all sentinel TIF files
            sentinel_files = [f for f in os.listdir(sentinel_dir) if f.endswith('.tif')]
            
            # Extract dates from filenames: sentinel2_site_date_band.tif
            date_to_files = defaultdict(list)
            date_pattern = re.compile(r'sentinel2_\w+_(\d{4}-\d{2}-\d{2})_\w+.tif')
            
            for file in sentinel_files:
                match = date_pattern.match(file)
                if match:
                    sentinel_date_str = match.group(1)
                    date_to_files[sentinel_date_str].append(file)
                else:
                    print(f"Warning: Could not extract date from filename: {file}")
            
            # Find the closest date
            closest_date_str = None
            min_days_diff = float('inf')
            
            for sentinel_date_str in date_to_files.keys():
                try:
                    sentinel_date = datetime.datetime.strptime(sentinel_date_str, '%Y-%m-%d')
                    days_diff = abs((sentinel_date - landsat_date).total_seconds() / (24 * 3600))
                    
                    if days_diff < min_days_diff:
                        min_days_diff = days_diff
                        closest_date_str = sentinel_date_str
                except ValueError:
                    print(f"Warning: Invalid date format {sentinel_date_str}, skipping")
            
            if closest_date_str is None:
                print(f"Warning: No valid sentinel dates found for {site}/{landsat_date_str}")
                continue
            
            # Keep files with the closest date, remove others
            for sentinel_date_str, files in date_to_files.items():
                if sentinel_date_str == closest_date_str:
                    # Keep these files
                    total_files_kept += len(files)
                    print(f"Keeping {len(files)} files for {site}/{landsat_date_str} with date {closest_date_str} (diff: {min_days_diff:.2f} days)")
                else:
                    # Remove these files
                    for file in files:
                        file_path = os.path.join(sentinel_dir, file)
                        try:
                            os.remove(file_path)
                            total_files_removed += 1
                        except Exception as e:
                            print(f"Error removing file {file_path}: {e}")
    
    print(f"\nCleaning complete!")
    print(f"Total date folders processed: {total_dates_processed}")
    print(f"Total Sentinel files kept: {total_files_kept}")
    print(f"Total Sentinel files removed: {total_files_removed}")

if __name__ == "__main__":
    base_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data'
    clean_sentinel_files_by_date(base_dir)