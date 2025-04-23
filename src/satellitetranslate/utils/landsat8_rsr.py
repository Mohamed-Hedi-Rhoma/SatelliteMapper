import numpy as np

class Landsat8RSRReader:
    def __init__(self, file_path):
        """
        Initialize the Landsat 8 RSR reader with a file path.
        
        Parameters:
        -----------
        file_path : str
            Path to the Landsat 8 RSR file
        """
        self.file_path = file_path
        
    def read_rsr_file(self):
        """
        Read the Landsat 8 RSR file and extract a matrix with the 6 bands.
        Rows with all zeros across the bands are removed.
        
        Returns:
        --------
        matrix : numpy.ndarray
            Matrix of shape [6, M] with band sensitivities
            where M is the number of non-zero wavelengths in the file
            The bands are in order: Blue, Green, Red, NIR, SWIR1, SWIR2
        wavelengths : numpy.ndarray
            Array of shape [M] with the wavelengths in micrometers
        """
        # Read all data from the file
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 9:  # Ensure we have enough columns
                    data.append([float(v) for v in values])
        
        # Convert to numpy array for easier manipulation
        data_array = np.array(data)
        
        # Define the column indices for each band based on the analysis
        # Column indices are 0-based in the array
        # Based on analysis, the bands appear to be in these columns:
        # Blue: column 2, Green: column 3, Red: column 4, 
        # NIR: column 5, SWIR1: column 6, SWIR2: column 7
        band_indices = [2, 3, 4, 5, 6, 7]
        
        # Extract wavelengths (first column)
        wavelengths = data_array[:, 0]
        
        # Create a temporary matrix to check which rows have non-zero values
        temp_matrix = np.zeros((6, len(data_array)))
        for i, band_idx in enumerate(band_indices):
            temp_matrix[i] = data_array[:, band_idx]
        
        # Find rows that have at least one non-zero value across any band
        non_zero_rows = np.any(temp_matrix > 0, axis=0)
        
        # Filter data_array and wavelengths to keep only non-zero rows
        filtered_data = data_array[non_zero_rows]
        filtered_wavelengths = wavelengths[non_zero_rows]
        
        # Count the number of filtered wavelengths
        n_wavelengths = len(filtered_wavelengths)
        
        # Create the output matrix [6, N] with only non-zero rows
        matrix = np.zeros((6, n_wavelengths))
        
        # Extract values for each band from the filtered data
        for i, band_idx in enumerate(band_indices):
            matrix[i] = filtered_data[:, band_idx]
        
        return matrix, filtered_wavelengths
    
    def get_band_names(self):
        """
        Returns the names of the bands in the order they appear in the output matrix.
        
        Returns:
        --------
        band_names : list
            List of band names
        """
        return ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

# Example usage
if __name__ == "__main__":
    rsr_reader = Landsat8RSRReader("/home/mrhouma/Documents/Project_perso/SatelliteMapper/data/L8_OLI_RSR.rsr")
    band_matrix, wavelengths = rsr_reader.read_rsr_file()
    
    # Print the shape of the resulting matrix
    print(f"Band matrix shape: {band_matrix.shape}")
    print(f"Wavelengths shape: {wavelengths.shape}")
    print(f"First few values of each band:")
    for i, band_name in enumerate(['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']):
        print(f"{band_name}: {band_matrix[i, :5]}")
    print(f"First five wavelengths: {wavelengths[:5]} micrometers")