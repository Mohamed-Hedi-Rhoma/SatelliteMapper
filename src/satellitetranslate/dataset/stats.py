import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from tqdm import tqdm

# Define directories
DATA_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data_prepared"
STATS_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/stats"

# Define bands to analyze
BANDS = ['blue.tif', 'green.tif', 'red.tif', 'nir.tif', 'swir1.tif', 'swir2.tif']
SATELLITES = ['sentinel2', 'landsat8']

def collect_band_data(satellite):
    """
    Collect all data for a specific satellite and band across all sites.
    
    Args:
        satellite (str): Satellite name ('sentinel2' or 'landsat8')
    
    Returns:
        dict: Dictionary with band data (key: band name, value: array of all valid pixels)
    """
    print(f"Collecting data for {satellite}...")
    
    # Dictionary to store all band data
    all_band_data = {band: [] for band in BANDS}
    
    # Get all site directories
    site_dirs = [d for d in glob.glob(os.path.join(DATA_DIR, "*")) 
                if os.path.isdir(d) and not d.endswith("__pycache__")]
    
    # Sample size for very large datasets (to avoid memory issues)
    max_samples_per_site = 100000  # Adjust based on your memory constraints
    
    # Process each site
    for site_dir in tqdm(site_dirs, desc=f"Processing sites for {satellite}"):
        # Get all acquisition date directories
        acq_dirs = [d for d in glob.glob(os.path.join(site_dir, "*")) 
                   if os.path.isdir(d) and not d.endswith("__pycache__")]
        
        # Process each acquisition date
        for acq_dir in acq_dirs:
            # Path to satellite directory
            sat_dir = os.path.join(acq_dir, satellite)
            
            if os.path.exists(sat_dir):
                # Process each band
                for band in BANDS:
                    band_path = os.path.join(sat_dir, band)
                    
                    if os.path.exists(band_path):
                        try:
                            with rasterio.open(band_path) as src:
                                # Read data
                                data = src.read(1)
                                
                                # Get valid data (exclude zeros and no-data values)
                                valid_data = data[data > 0]
                                
                                if len(valid_data) > 0:
                                    # If there's a lot of data, take a random sample
                                    if len(valid_data) > max_samples_per_site:
                                        indices = np.random.choice(len(valid_data), max_samples_per_site, replace=False)
                                        valid_data = valid_data[indices]
                                    
                                    # Append to the list for this band
                                    all_band_data[band].append(valid_data)
                        except Exception as e:
                            print(f"Error reading {band_path}: {e}")
    
    # Combine all data for each band
    for band in BANDS:
        if all_band_data[band]:
            # Concatenate all arrays for this band
            all_band_data[band] = np.concatenate(all_band_data[band])
        else:
            all_band_data[band] = np.array([])
    
    return all_band_data

def calculate_statistics(band_data):
    """
    Calculate statistics for band data.
    
    Args:
        band_data (dict): Dictionary with band data
    
    Returns:
        dict: Dictionary with statistics for each band
    """
    stats = {}
    
    for band, data in band_data.items():
        if len(data) > 0:
            # Calculate statistics
            stats[band] = {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'p1': float(np.percentile(data, 1)),
                'p5': float(np.percentile(data, 5)),
                'p25': float(np.percentile(data, 25)),
                'p50': float(np.percentile(data, 50)),
                'p75': float(np.percentile(data, 75)),
                'p95': float(np.percentile(data, 95)),
                'p99': float(np.percentile(data, 99)),
                'count': int(len(data))
            }
            
            # Calculate histogram for plotting
            hist, bin_edges = np.histogram(data, bins=100, range=(0, 1))
            stats[band]['histogram'] = hist.tolist()
            stats[band]['bin_edges'] = bin_edges.tolist()
        else:
            stats[band] = {
                'min': None,
                'max': None,
                'mean': None,
                'std': None,
                'p1': None,
                'p5': None,
                'p25': None,
                'p50': None,
                'p75': None,
                'p95': None,
                'p99': None,
                'count': 0,
                'histogram': None,
                'bin_edges': None
            }
    
    return stats

def generate_histogram_plots(stats, satellite):
    """
    Generate histogram plots for each band.
    
    Args:
        stats (dict): Statistics dictionary
        satellite (str): Satellite name
        
    Returns:
        str: Path to saved plot file
    """
    # Create directory for plots
    os.makedirs(os.path.join(STATS_DIR, "plots"), exist_ok=True)
    
    # Create a figure with subplots for each band
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f"{satellite} Band Histograms (All Sites)", fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot histograms for each band
    for i, band in enumerate(BANDS):
        if stats[band]['histogram'] is not None:
            # Get histogram data
            hist = np.array(stats[band]['histogram'])
            bin_edges = np.array(stats[band]['bin_edges'])
            
            # Create bin centers for plotting
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            # Plot histogram
            axes[i].bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), 
                       alpha=0.7, color='steelblue', edgecolor='black')
            
            # Add statistics as text
            stats_text = (
                f"Min: {stats[band]['min']:.4f}\n"
                f"Max: {stats[band]['max']:.4f}\n"
                f"Mean: {stats[band]['mean']:.4f}\n"
                f"Std: {stats[band]['std']:.4f}\n"
                f"P5: {stats[band]['p5']:.4f}\n"
                f"P95: {stats[band]['p95']:.4f}\n"
                f"Count: {stats[band]['count']:,}"
            )
            axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Set labels and title
            axes[i].set_xlabel('Reflectance Value')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f"{band.replace('.tif', '')} Band")
            
            # Set x axis limits
            axes[i].set_xlim(0, 1)
        else:
            axes[i].text(0.5, 0.5, "No data available", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[i].transAxes)
            axes[i].set_title(f"{band.replace('.tif', '')} Band")
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    # Save the figure
    save_path = os.path.join(STATS_DIR, "plots", f"{satellite}_histograms.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    return save_path

def generate_band_comparison_plot(all_stats):
    """
    Generate a bar plot comparing mean values for each band across satellites.
    
    Args:
        all_stats (dict): Dictionary with statistics for each satellite
    
    Returns:
        str: Path to saved plot file
    """
    # Create directory for plots
    os.makedirs(os.path.join(STATS_DIR, "plots"), exist_ok=True)
    
    # Extract mean values for each band and satellite
    means = defaultdict(dict)
    
    for satellite in SATELLITES:
        if satellite in all_stats:
            for band in BANDS:
                if band in all_stats[satellite] and all_stats[satellite][band]['mean'] is not None:
                    means[band][satellite] = all_stats[satellite][band]['mean']
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.35
    x = np.arange(len(BANDS))
    
    # Plot bars for each satellite
    for i, satellite in enumerate(SATELLITES):
        satellite_means = [means[band].get(satellite, 0) for band in BANDS]
        ax.bar(x + i*bar_width - bar_width/2, satellite_means, bar_width, 
               label=satellite, alpha=0.7)
    
    # Add labels and legend
    ax.set_xlabel('Band')
    ax.set_ylabel('Mean Reflectance')
    ax.set_title('Mean Reflectance by Band and Satellite')
    ax.set_xticks(x)
    ax.set_xticklabels([band.replace('.tif', '') for band in BANDS])
    ax.legend()
    
    # Save the figure
    save_path = os.path.join(STATS_DIR, "plots", "band_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    return save_path

def create_csv_summary(all_stats):
    """
    Create a CSV summary of all statistics.
    
    Args:
        all_stats (dict): Dictionary with statistics for each satellite
    """
    # Create directory for CSV files
    os.makedirs(os.path.join(STATS_DIR, "summary"), exist_ok=True)
    
    # Create a DataFrame for each satellite
    for satellite in SATELLITES:
        if satellite in all_stats:
            # Initialize lists for DataFrame
            rows = []
            
            # Add each band's statistics
            for band in BANDS:
                if band in all_stats[satellite] and all_stats[satellite][band]['mean'] is not None:
                    stats = all_stats[satellite][band]
                    rows.append({
                        'Band': band.replace('.tif', ''),
                        'Min': stats['min'],
                        'Max': stats['max'],
                        'Mean': stats['mean'],
                        'Std': stats['std'],
                        'P5': stats['p5'],
                        'P50': stats['p50'],
                        'P95': stats['p95'],
                        'Count': stats['count']
                    })
            
            # Create and save DataFrame
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(STATS_DIR, "summary", f"{satellite}_statistics.csv"), index=False)

def main():
    """
    Main function to collect statistics and generate visualizations.
    """
    # Create stats directory if it doesn't exist
    os.makedirs(STATS_DIR, exist_ok=True)
    
    all_stats = {}
    
    # Process each satellite
    for satellite in SATELLITES:
        # Collect data for all bands
        band_data = collect_band_data(satellite)
        
        # Calculate statistics
        print(f"Calculating statistics for {satellite}...")
        stats = calculate_statistics(band_data)
        all_stats[satellite] = stats
        
        # Generate histogram plots
        print(f"Generating histograms for {satellite}...")
        hist_path = generate_histogram_plots(stats, satellite)
        print(f"Saved histograms to {hist_path}")
        
        # Save statistics to JSON
        stats_path = os.path.join(STATS_DIR, f"{satellite}_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")
    
    # Generate band comparison plot
    print("Generating band comparison plot...")
    comparison_path = generate_band_comparison_plot(all_stats)
    print(f"Saved band comparison to {comparison_path}")
    
    # Create CSV summary
    print("Creating CSV summary...")
    create_csv_summary(all_stats)
    
    print("Statistics generation completed successfully!")

if __name__ == "__main__":
    main()